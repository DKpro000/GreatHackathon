import os, re, json, math, glob, faiss, boto3, numpy as np, pandas as pd, threading, time, random, hashlib, pickle, concurrent.futures
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import lru_cache
from botocore.exceptions import ClientError
import difflib
from botocore.config import Config
from collections import Counter

from events_store import add_event, get_recent_events, init_db as _init_events_db
from personalize import (
    build_user_profile, rerank, budget_fit, decay_weight,
)

ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR", "archive")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

def safe_int(v, d): 
    try: return int(v)
    except: return d

def safe_float(v, d):
    try: return float(v)
    except: return d

TOPK_CANDIDATES = safe_int(os.environ.get("TOPK_CANDIDATES", "200"), 200)  # 候选↑，给重排空间
TOPK_RETURN     = safe_int(os.environ.get("TOPK_RETURN", "50"), 50)
BUDGET_TOLERANCE = safe_float(os.environ.get("BUDGET_TOLERANCE", "0.15"), 0.15)
MAX_WORKERS      = safe_int(os.environ.get("MAX_WORKERS", "2"), 2)
CACHE_DIR        = os.environ.get("CACHE_DIR", "cache")
RATE_LIMIT_DELAY = safe_float(os.environ.get("RATE_LIMIT_DELAY", "0.5"), 0.5)
USE_SIMPLE_EMBED_QUERY_ONLY = os.environ.get("USE_SIMPLE_EMBED_QUERY_ONLY", "0") == "1"
SIMPLE_DIM = 1536

EXRATE = {
    "MYR": float(os.environ.get("EXRATE_MYR_MYR", "1.0")),
    "RM":  safe_float(os.environ.get("EXRATE_RM_MYR", "1.0"), 1.0),
    "INR": safe_float(os.environ.get("EXRATE_INR_MYR", "0.056"), 0.056),
    "₹":   safe_float(os.environ.get("EXRATE_INR_MYR", "0.056"), 0.056),
    "RS":  safe_float(os.environ.get("EXRATE_INR_MYR", "0.056"), 0.056),
    "USD": safe_float(os.environ.get("EXRATE_USD_MYR", "4.2"), 4.2),
    "$":   safe_float(os.environ.get("EXRATE_USD_MYR", "4.2"), 4.2),
    "SGD": safe_float(os.environ.get("EXRATE_SGD_MYR", "3.5"), 3.5),
    "CNY": safe_float(os.environ.get("EXRATE_CNY_MYR", "0.65"), 0.65),
}

def create_bedrock_client():
    try:
        cfg = Config(
            connect_timeout=3,
            read_timeout=5,
            retries={"max_attempts": 2, "mode": "standard"},
        )
        return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION, config=cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to create Bedrock client: {e}")

client = None
client_lock = threading.Lock()
def get_client():
    global client
    if client is None:
        with client_lock:
            if client is None:
                client = create_bedrock_client()
    return client

rate_limit_lock = threading.Lock()
_last_req_time = 0.0
def wait_for_rate_limit():
    global _last_req_time
    with rate_limit_lock:
        now = time.time()
        if now - _last_req_time < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - (now - _last_req_time))
        _last_req_time = time.time()

def backoff(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except ClientError as e:
            code = e.response.get('Error',{}).get('Code','')
            if code == 'ThrottlingException' and i < tries-1:
                wt = (2**i) + random.uniform(0,1)
                time.sleep(wt); continue
            raise
        except Exception as e:
            if i < tries-1:
                wt = (2**i) + random.uniform(0,1)
                time.sleep(wt); continue
            raise

@lru_cache(maxsize=1000)
def get_embedding_cached(text: str) -> tuple:
    def _call():
        wait_for_rate_limit()
        c = get_client()
        body = json.dumps({"inputText": text})
        print(f"[bedrock] invoke start len={len(text)}")
        r = c.invoke_model(modelId=EMBED_MODEL_ID, accept="application/json",
                           contentType="application/json", body=body)
        payload = json.loads(r["body"].read())
        vec = payload.get("embedding") or payload.get("vector")
        if vec is None:
            raise RuntimeError("Bedrock response missing 'embedding'")
        print(f"[bedrock] invoke ok")
        return tuple(vec)
    return backoff(_call)

def get_embedding(text: str) -> np.ndarray:
    return np.array(get_embedding_cached(text), dtype="float32")

def get_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    out = []
    bs = min(MAX_WORKERS, 10)
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batch))) as ex:
            futs = [ex.submit(get_embedding, t) for t in batch]
            for f in concurrent.futures.as_completed(futs):
                try: out.append(f.result())
                except: out.append(np.zeros(1536, dtype="float32"))
        if i + bs < len(texts): time.sleep(2)
    return out

import pandas as pd
PRICE_PAT = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)")

def detect_currency_symbol(s: str) -> str:
    if not s: return "MYR"
    u = s.strip().upper()
    if "₹" in s or "INR" in u or "RS" in u: return "INR"
    if u.startswith("RM") or "MYR" in u: return "MYR"
    if "$" in s or "USD" in u: return "USD"
    if "SGD" in u or "S$" in u: return "SGD"
    if "CNY" in u or "RMB" in u or "¥" in s: return "CNY"
    return "MYR"

def parse_price_to_myr(raw):
    if raw is None or (isinstance(raw,float) and math.isnan(raw)): return None
    s = str(raw); cur = detect_currency_symbol(s)
    m = PRICE_PAT.search(s.replace(" ",""))
    if not m: return None
    num = float(m.group(1).replace(",",""))
    return round(num * EXRATE.get(cur,1.0), 2)

def coalesce_price(row: dict):
    for k in ("discount_price","Discount Price","discount","price","Price","actual_price","Actual Price"):
        if k in row and pd.notna(row[k]):
            p = parse_price_to_myr(row[k])
            if p is not None: return p
    return None

READ_COL_MAP = {
  "name":"name","title":"name","product_name":"name",
  "main_category":"main_category","category":"main_category",
  "sub_category":"sub_category","image":"image",
  "link":"link","url":"link",
  "ratings":"ratings","rating":"ratings",
  "no_of_ratings":"no_of_ratings","no_of_reviews":"no_of_ratings",
  "discount_price":"discount_price","actual_price":"actual_price"
}
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: READ_COL_MAP.get(str(c).strip(), c) for c in df.columns})

def read_any_table(path: str) -> pd.DataFrame:
    try:
        if path.lower().endswith(".csv"): df = pd.read_csv(path)
        elif path.lower().endswith(".xlsx"): df = pd.read_excel(path, engine="openpyxl")
        else: df = pd.read_excel(path, engine="xlrd")
    except FileNotFoundError: raise
    except pd.errors.EmptyDataError: return pd.DataFrame()
    except Exception:
        try: df = pd.read_excel(path)
        except: return pd.DataFrame()
    df = normalize_columns(df)
    keep = ["name","main_category","sub_category","image","link","ratings","no_of_ratings","discount_price","actual_price"]
    for k in keep:
        if k not in df.columns: df[k] = None
    return df[keep]

def load_products_from_file(path: str) -> list[dict]:
    try:
        if path.lower().endswith(".csv"): df = pd.read_csv(path)
        else: df = read_any_table(path)
        if df.empty: return []
        name_col = "title" if "title" in df.columns else "name"
        if name_col not in df.columns: return []
        df = df.dropna(subset=[name_col])

        if "category_id" in df.columns: limited = df.groupby("category_id").head(20)
        elif "main_category" in df.columns: limited = df.groupby("main_category").head(20)
        else: limited = df.head(20)

        items = []
        for idx, r in limited.iterrows():
            row = r.to_dict()
            category_id = None
            raw_cid = row.get("category_id")
            if raw_cid is not None and not (isinstance(raw_cid, float) and math.isnan(raw_cid)):
                category_id = _normalize_category_id(raw_cid)
            if (not row.get("main_category") or str(row.get("main_category")).strip() == "") and category_id:
                mapped_name = CATEGORY_BY_ID.get(category_id)
                if mapped_name:
                    row["main_category"] = mapped_name
            elif row.get("main_category") and category_id is None:
                mapped_name = CATEGORY_NAME_TO_ID.get(str(row.get("main_category")).strip().lower())
                if mapped_name:
                    category_id = mapped_name
            price_myr = None
            if "price(usd)" in row and pd.notna(row["price(usd)"]):
                try: price_myr = float(row["price(usd)"]) * EXRATE.get("USD",4.2)
                except: pass
            if price_myr is None: price_myr = coalesce_price(row)
            parts = [str(row.get(name_col,""))]
            if row.get("main_category"): parts.append(str(row.get("main_category")))
            if row.get("sub_category"): parts.append(str(row.get("sub_category")))
            text = " ".join(parts).strip()
            items.append({
                "id": f"{os.path.basename(path)}::{idx}",
                "name": row.get(name_col,""),
                "category_id": category_id,
                "main_category": row.get("main_category",""),
                "sub_category":  row.get("sub_category",""),
                "image": row.get("image") or row.get("imgUrl",""),
                "link":  row.get("link") or row.get("productURL",""),
                "ratings": row.get("ratings") or row.get("stars",""),
                "no_of_ratings": row.get("no_of_ratings") or row.get("reviews",""),
                "discount_price": row.get("discount_price") or row.get("price(usd)",""),
                "actual_price": row.get("actual_price") or row.get("listPrice",""),
                "price_myr": price_myr,
                "doc_path": path,
                "text": text,
            })
        return items
    except Exception:
        return []

def load_all_products(archive_dir: str) -> list[dict]:
    if not os.path.exists(archive_dir):
        raise FileNotFoundError(f"Archive not found: {archive_dir}")
    paths = sorted(
        glob.glob(os.path.join(archive_dir,"**","*.csv"), recursive=True) +
        glob.glob(os.path.join(archive_dir,"**","*.xls"), recursive=True) +
        glob.glob(os.path.join(archive_dir,"**","*.xlsx"), recursive=True)
    )
    if not paths:
        raise FileNotFoundError(f"No CSV/XLS under {archive_dir}")
    all_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut2p = {ex.submit(load_products_from_file, p): p for p in paths}
        for fut in concurrent.futures.as_completed(fut2p):
            try: all_items.extend(fut.result())
            except: pass
    return all_items

def _corpus_hash(texts: list[str]) -> str:
    h = hashlib.md5()
    for t in texts: h.update((t[:200]+"\n").encode("utf-8","ignore"))
    return h.hexdigest()

def _save_cache(arr: np.ndarray, path: str, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"wb") as f:
        pickle.dump({"vectors": np.asarray(arr, dtype="float32"), "meta": meta}, f)

def _load_cache(path: str):
    try:
        with open(path,"rb") as f: payload = pickle.load(f)
    except Exception: return None
    vecs = payload.get("vectors")
    if isinstance(vecs, list): vecs = np.array(vecs, dtype="float32")
    elif isinstance(vecs, np.ndarray): vecs = vecs.astype("float32", copy=False)
    else: return None
    return {"vectors": vecs, "meta": payload.get("meta", {})}

def build_faiss(products: list[dict]):
    corpus = [p["text"] for p in products]
    if not corpus: raise RuntimeError("No products to index")
    cache_path = os.path.join(CACHE_DIR, "embeddings.pkl")
    chash = _corpus_hash(corpus); exp_count = len(corpus)
    cached = _load_cache(cache_path)
    X = None
    if cached:
        vecs = cached["vectors"]; meta = cached.get("meta",{})
        if isinstance(vecs,np.ndarray) and vecs.ndim==2 and vecs.shape[0]==exp_count and \
           meta.get("corpus_hash")==chash and meta.get("model_id")==EMBED_MODEL_ID:
            X = vecs
        else:
            print("Embedding cache invalid -> regenerate")
    if X is None:
        vectors = get_embeddings_batch(corpus)
        X = np.vstack([np.asarray(v, dtype="float32") for v in vectors])
        _save_cache(X, cache_path, {"model_id": EMBED_MODEL_ID, "corpus_hash": chash, "count": exp_count, "saved_at": datetime.now().isoformat()})
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    Xn = X.copy(); faiss.normalize_L2(Xn)
    index.add(Xn)
    return index, X


def get_product_vector(pid: str):
    if VECS is None:
        return None
    idx = ID_TO_INDEX.get(pid)
    if idx is None:
        return None
    return VECS[idx]

def get_simple_query_vec(text: str, top_n: int = 50) -> np.ndarray:
    if VECS is None or len(VECS) == 0:
        raise RuntimeError("Vector index not ready")
    toks = _tokenize(text)
    with TOKEN_INDEX_LOCK:
        mapping = TOKEN_TO_IDS
    scores: dict[int, int] = {}
    for tok in toks:
        for idx in mapping.get(tok, []):
            scores[idx] = scores.get(idx, 0) + 1
    if not scores and VECS.size:
        avg = VECS.mean(axis=0)
    else:
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n] if scores else []
        if not ordered:
            avg = VECS.mean(axis=0)
        else:
            indices = [idx for idx, _ in ordered]
            weights = np.array([cnt for _, cnt in ordered], dtype="float32")
            vecs = VECS[indices]
            avg = np.average(vecs, axis=0, weights=weights) if len(indices) > 1 else vecs[0]
    norm = float(np.linalg.norm(avg) + 1e-12)
    return (avg / norm).astype("float32")

VOCAB = set(); VOCAB_LOCK = threading.Lock()
def _tokenize(text: str): return re.findall(r"[a-zA-Z0-9\u4e00-\u9fa5]+", (text or "").lower())
def rebuild_vocab(products: list[dict]):
    global VOCAB
    words = set()
    for p in products:
        blob = f"{p.get('name','')} {p.get('main_category','')} {p.get('sub_category','')}"
        for t in _tokenize(blob):
            if len(t) >= 2: words.add(t)
    with VOCAB_LOCK: VOCAB = words

def vocab_contains(tok: str) -> bool:
    with VOCAB_LOCK: return tok in VOCAB

def suggest_token(token: str, n=1, cutoff=0.82):
    with VOCAB_LOCK: 
        return difflib.get_close_matches(token, VOCAB, n=n, cutoff=cutoff)

def expand_query_typos(q: str) -> dict:
    toks = _tokenize(q); fixed = []; tokens_map=[]
    for t in toks:
        if vocab_contains(t) or len(t)<3:
            fixed.append(t); tokens_map.append((t,[])); continue
        sug = suggest_token(t,1,0.82)
        if sug: fixed.append(sug[0]); tokens_map.append((t,sug))
        else:   fixed.append(t); tokens_map.append((t,[]))
    return {"original": q, "corrected": " ".join(fixed), "tokens_map": tokens_map}

BUDGET_PAT = re.compile(r"(?P<amt>\d[\d,]*(?:\.\d+)?)\s*(?P<cur>RM|MYR|马币|INR|₹|USD|\$|SGD|CNY|RMB|¥)?", re.I)
def extract_budget_myr(q: str):
    if not q: return None
    hits = list(BUDGET_PAT.finditer(q))
    if not hits: return None
    m = hits[-1]
    try: amt = float(m.group("amt").replace(",",""))
    except: return None
    cur_raw = (m.group("cur") or "").upper()
    cur = "MYR" if (("马币" in q) or ("令吉" in q) or cur_raw=="") else cur_raw
    return round(amt * EXRATE.get(cur,1.0), 2)



def parse_price_range(raw) -> tuple[float | None, float | None] | None:
    if raw is None:
        return None
    lo_raw = hi_raw = None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        if '-' in raw:
            lo_raw, hi_raw = raw.split('-', 1)
        else:
            lo_raw, hi_raw = raw, None
    elif isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            return None
        lo_raw, hi_raw = raw
    else:
        lo_raw, hi_raw = raw, None

    def to_float(val):
        if val is None:
            return None
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return None
        try:
            v = float(val)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        return v

    lo = to_float(lo_raw)
    hi = to_float(hi_raw)
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None and hi < lo:
        lo, hi = hi, lo
    return (lo, hi)


def price_range_to_list(r: tuple[float | None, float | None] | None) -> list[float | None] | None:
    if not r:
        return None
    lo, hi = r
    return [lo, hi]

CATEGORY_HINTS = {
    "laptop":["laptop","notebook","笔记本","电脑","macbook"],
    "desktop":["台式机","主机","desktop","pc"],
    "phone":["手机","phone","智能手机","iphone","android"],
    "camera":["相机","camera","单反","微单"],
    "shoes":["鞋","shoes","sneaker"],
}


CATEGORIES_FILE = os.environ.get("CATEGORIES_FILE", os.path.join(ARCHIVE_DIR, "amazon_categories.csv"))
CATEGORY_BY_ID: dict[str, str] = {}
CATEGORY_NAME_TO_ID: dict[str, str] = {}
CATEGORY_KEYWORD_INDEX: dict[str, set[str]] = {}
GLOBAL_CATEGORY_COUNTS: Counter = Counter()
GLOBAL_SUBCATEGORY_COUNTS: Counter = Counter()
CATEGORY_LOCK = threading.Lock()


def _normalize_category_id(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        value = int(value) if float(value).is_integer() else value
    s = str(value).strip()
    if not s:
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_categories():
    global CATEGORY_BY_ID, CATEGORY_NAME_TO_ID, CATEGORY_KEYWORD_INDEX
    path = CATEGORIES_FILE
    if not path:
        return
    if not os.path.exists(path):
        CATEGORY_BY_ID = {}
        CATEGORY_NAME_TO_ID = {}
        CATEGORY_KEYWORD_INDEX = {}
        return
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        CATEGORY_BY_ID = {}
        CATEGORY_NAME_TO_ID = {}
        CATEGORY_KEYWORD_INDEX = {}
        return

    id_col = None
    name_col = None
    for cand in ("id", "category_id", "categoryId", "categoryID"):
        if cand in df.columns:
            id_col = cand
            break
    for cand in ("category_name", "Category", "name", "categoryName"):
        if cand in df.columns:
            name_col = cand
            break

    if id_col is None or name_col is None:
        CATEGORY_BY_ID = {}
        CATEGORY_NAME_TO_ID = {}
        CATEGORY_KEYWORD_INDEX = {}
        return

    mapping: dict[str, str] = {}
    name_map: dict[str, str] = {}
    kw_index: dict[str, set[str]] = {}

    for _, row in df[[id_col, name_col]].dropna().iterrows():
        cid = _normalize_category_id(row[id_col])
        if not cid:
            continue
        name = str(row[name_col]).strip()
        if not name:
            continue
        mapping[cid] = name
        name_map[name.lower()] = cid
        tokens = set(_tokenize(name))
        if not tokens:
            continue
        for tok in tokens:
            kw_index.setdefault(tok, set()).add(cid)

    CATEGORY_BY_ID = mapping
    CATEGORY_NAME_TO_ID = name_map
    CATEGORY_KEYWORD_INDEX = kw_index


def infer_category_ids_from_tokens(tokens: set[str], limit: int = 5) -> list[str]:
    scores: dict[str, int] = {}
    for tok in tokens:
        for cid in CATEGORY_KEYWORD_INDEX.get(tok, () ):
            scores[cid] = scores.get(cid, 0) + 1
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [cid for cid, _ in ranked]


def infer_category_ids_from_query(q: str, limit: int = 5) -> list[str]:
    if not q:
        return []
    tokens = set(_tokenize(q))
    return infer_category_ids_from_tokens(tokens, limit)


def update_global_category_stats(products: list[dict]):
    global GLOBAL_CATEGORY_COUNTS, GLOBAL_SUBCATEGORY_COUNTS
    cat_counter = Counter()
    sub_counter = Counter()
    for prod in products:
        cat = prod.get("main_category")
        if cat:
            cat_counter[cat] += 1
        sub = prod.get("sub_category")
        if sub:
            sub_counter[sub] += 1
    GLOBAL_CATEGORY_COUNTS = cat_counter
    GLOBAL_SUBCATEGORY_COUNTS = sub_counter
def keywords_from_query(q: str):
    ql = (q or "").lower(); found=[]
    for key,ws in CATEGORY_HINTS.items():
        if any(w in ql for w in ws): found.append(key)
    toks = _tokenize(ql)
    return list(set(found + toks))

def fuzzy_contains(blob: str, token: str, cutoff=0.82) -> bool:
    words = _tokenize(blob)
    if token in words: return True
    for w in words:
        if difflib.SequenceMatcher(a=token,b=w).ratio() >= cutoff:
            return True
    return False

def passes_keywords(p: dict, kws: list[str]) -> bool:
    cat = [k for k in CATEGORY_HINTS if k in kws]
    if not cat: return True
    blob = f"{p.get('name','')} {p.get('main_category','')} {p.get('sub_category','')}".lower()
    strict = any(w in blob for w in sum((CATEGORY_HINTS[k] for k in cat), []))
    if strict: return True
    for k in cat:
        for w in CATEGORY_HINTS[k]:
            if fuzzy_contains(blob, w, 0.82): return True
    return False

PRODUCTS, INDEX, VECS = [], None, None
ID_TO_INDEX = {}
TOKEN_TO_IDS = {}
TOKEN_INDEX_LOCK = threading.Lock()


def rebuild_token_index(products: list[dict]):
    global TOKEN_TO_IDS
    mapping: dict[str, list[int]] = {}
    for idx, p in enumerate(products):
        tokens = set(_tokenize(p.get("text", "")))
        for tok in tokens:
            mapping.setdefault(tok, []).append(idx)
    with TOKEN_INDEX_LOCK:
        TOKEN_TO_IDS = mapping

def get_product_by_id(pid: str):
    if not pid:
        return None
    idx = ID_TO_INDEX.get(pid)
    if idx is None:
        return None
    try:
        return PRODUCTS[idx]
    except IndexError:
        return None

def ensure_index():
    global PRODUCTS, INDEX, VECS, ID_TO_INDEX
    load_categories()
    PRODUCTS = load_all_products(ARCHIVE_DIR)
    update_global_category_stats(PRODUCTS)
    INDEX, VECS = build_faiss(PRODUCTS)
    ID_TO_INDEX = {p["id"]: idx for idx, p in enumerate(PRODUCTS)}
    rebuild_vocab(PRODUCTS)
    rebuild_token_index(PRODUCTS)

def init_index_async():
    try:
        ensure_index()
        print(f"[{datetime.now().isoformat()}] Indexed items: {len(PRODUCTS)}")
    except Exception as e:
        print("Index init failed:", e)

threading.Thread(target=init_index_async, daemon=True).start()
_init_events_db()

app = Flask(__name__)
CORS(app)

@app.route("/event", methods=["POST"])
def track_event():
    data = request.get_json(force=True)
    user_id   = data.get("user_id") or request.headers.get("X-User-Id") or "anon"
    etype     = data["type"]
    query     = data.get("query")
    product_id= data.get("product_id")
    meta      = data.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    if etype == "click" and product_id:
        prod = get_product_by_id(product_id)
        if prod:
            meta = dict(meta)
            if prod.get("main_category") and "main_category" not in meta:
                meta["main_category"] = prod.get("main_category")
            if prod.get("sub_category") and "sub_category" not in meta:
                meta["sub_category"] = prod.get("sub_category")
            if prod.get("category_id") and "category_id" not in meta:
                meta["category_id"] = prod.get("category_id")
            price_val = prod.get("price_myr")
            if isinstance(price_val, (int, float)) and "price_myr" not in meta:
                meta["price_myr"] = float(price_val)
    add_event(user_id, etype, query=query, product_id=product_id, meta=meta)
    return jsonify({"ok": True})

@app.route("/search", methods=["GET","POST"])
def search():
    payload = {}
    price_range = None
    if request.method == "GET":
        q = (request.args.get("query") or "").strip()
        raw_price = request.args.get("price")
        price_range = parse_price_range(raw_price)
        if price_range is None:
            price_range = parse_price_range((request.args.get("price_min"), request.args.get("price_max")))
    else:
        payload = request.get_json(silent=True) or {}
        q = (payload.get("query") or "").strip()
        raw_price = payload.get("price")
        if raw_price is None and ("price_min" in payload or "price_max" in payload):
            raw_price = (payload.get("price_min"), payload.get("price_max"))
        price_range = parse_price_range(raw_price)

    if not q:
        return jsonify({"error":"Query parameter required"}), 400

    user_id = request.headers.get("X-User-Id","anon")
    opt_out = request.headers.get("X-Opt-Out","false").lower() == "true"

    if INDEX is None or not PRODUCTS:
        return jsonify({"error":"Index is still building, please try again shortly"}), 503

    expand = expand_query_typos(q)
    corrected = expand["corrected"]
    has_corr  = corrected != q
    budget_myr = extract_budget_myr(q)

    def product_in_budget(prod: dict) -> bool:
        if budget_myr is None:
            return True
        price = prod.get("price_myr")
        if price is None:
            return False
        lo = budget_myr * (1 - BUDGET_TOLERANCE)
        hi = budget_myr * (1 + BUDGET_TOLERANCE)
        return lo <= price <= hi

    def product_in_price_range(prod: dict) -> bool:
        if price_range is None:
            return True
        price = prod.get("price_myr")
        if price is None:
            return False
        lo, hi = price_range
        if lo is not None and price < lo:
            return False
        if hi is not None and price > hi:
            return False
        return True

    fallback = False
    chosen_ids = []
    embedding_used = "bedrock"
    embed_simple_used = USE_SIMPLE_EMBED_QUERY_ONLY
    try:
        embed_simple_used = False
        embedding_mode = "bedrock"

        def embed_text(value: str) -> np.ndarray:
            nonlocal embed_simple_used, embedding_mode
            if USE_SIMPLE_EMBED_QUERY_ONLY or embed_simple_used:
                embed_simple_used = True
                embedding_mode = "simple"
                return get_simple_query_vec(value)
            try:
                return get_embedding(value)
            except Exception:
                embed_simple_used = True
                embedding_mode = "simple"
                return get_simple_query_vec(value)

        qv_orig = embed_text(q).reshape(1, -1).astype("float32")

        if has_corr:
            qv_corr = embed_text(corrected).reshape(1, -1).astype("float32")
            qv = (qv_orig * 1.0 + qv_corr * 0.9) / 1.9
        else:
            qv = qv_orig

        faiss.normalize_L2(qv)

        kws = _tokenize(corrected if has_corr else q)
        if has_corr:
            kws.extend(_tokenize(q))
        if len(kws) > 12:
            kws = kws[:12]

        cand_ids = []
        _, I = INDEX.search(qv, TOPK_CANDIDATES)
        for idx in I[0]:
            p = PRODUCTS[idx]
            if not passes_keywords(p, kws): continue
            if not product_in_budget(p): continue
            if not product_in_price_range(p): continue
            cand_ids.append(p["id"])

        if not cand_ids:
            for idx in I[0][:TOPK_RETURN * 3]:
                p = PRODUCTS[idx]
                if not passes_keywords(p, kws): continue
                if not product_in_price_range(p): continue
                cand_ids.append(p["id"])

        if not opt_out:
            id_to_product = {p["id"]: p for p in PRODUCTS}

            def get_prod_vec(pid):
                vec = get_product_vector(pid)
                if vec is not None:
                    return vec
                prod = id_to_product.get(pid)
                if not prod:
                    return None
                return get_simple_query_vec(prod.get("text", ""))

            def get_query_vec(text_value):
                return embed_text(text_value)

            def get_product_meta(pid):
                prod = id_to_product.get(pid, {})
                return {
                    "main_category": prod.get("main_category"),
                    "sub_category":  prod.get("sub_category"),
                    "brand":         prod.get("sub_category"),
                    "price_myr":     prod.get("price_myr"),
                }

            profile = build_user_profile(user_id, get_prod_vec, get_query_vec, get_product_meta)
            qvec = qv[0]
            ranked = rerank(qvec, profile, cand_ids, get_prod_vec, get_product_meta)
            chosen_ids = [pid for pid, _ in ranked[:TOPK_RETURN]]
        else:
            chosen_ids = cand_ids[:TOPK_RETURN]

        embedding_used = embedding_mode
    except Exception as e:
        fallback = True
        embedding_used = "keyword"
        toks = _tokenize(corrected if has_corr else q)
        def kw_score(p):
            blob = f"{p.get('name','')} {p.get('main_category','')} {p.get('sub_category','')}".lower()
            strict = sum(1 for t in toks if t in blob)
            fuzzy  = sum(1 for t in toks if fuzzy_contains(blob, t, 0.82))
            base = strict*1.0 + fuzzy*0.6
            price = p.get("price_myr")
            if budget_myr and price:
                base *= budget_fit(budget_myr, price)
            return base
        scored = []
        for p in PRODUCTS:
            if not product_in_price_range(p):
                continue
            s = kw_score(p)
            if s>0: scored.append((s,p["id"]))
        scored.sort(key=lambda x:x[0], reverse=True)
        chosen_ids = [pid for _,pid in scored[:TOPK_RETURN]]

    id2p = {p["id"]: p for p in PRODUCTS}
    results = []
    for pid in chosen_ids:
        p = id2p.get(pid)
        if not p:
            continue
        results.append({
            "id": p["id"], "name": p["name"],
            "main_category": p["main_category"], "sub_category": p["sub_category"],
            "image": p["image"], "link": p["link"],
            "ratings": p["ratings"], "no_of_ratings": p["no_of_ratings"],
            "discount_price": p["discount_price"], "actual_price": p["actual_price"],
            "price_myr": p["price_myr"], "source_file": p["doc_path"],
        })


    cat_counter = Counter()
    sub_counter = Counter()
    price_values: list[float] = []
    for item in results:
        cat_val = item.get("main_category")
        if cat_val:
            cat_counter[cat_val] += 1
        sub_val = item.get("sub_category")
        if sub_val:
            sub_counter[sub_val] += 1
        price_val = item.get("price_myr")
        if isinstance(price_val, (int, float)):
            price_values.append(float(price_val))

    top_categories = [name for name, _ in cat_counter.most_common(5)]
    top_subcategories = [name for name, _ in sub_counter.most_common(5)]
    result_price_range = None
    avg_price = None
    if price_values:
        result_price_range = [round(min(price_values), 2), round(max(price_values), 2)]
        avg_price = round(sum(price_values) / len(price_values), 2)

    category_ids = []
    for name in top_categories:
        if isinstance(name, str):
            cid = CATEGORY_NAME_TO_ID.get(name.strip().lower())
            if cid:
                category_ids.append(cid)

    search_meta = {
        "categories": top_categories,
        "subcategories": top_subcategories,
        "category_ids": category_ids,
        "price_range": result_price_range,
        "avg_price": avg_price,
        "applied_price_filter": price_range_to_list(price_range),
        "result_count": len(results),
    }
    search_meta = {k: v for k, v in search_meta.items() if v not in (None, [], {})}

    try:
        add_event(user_id, "search", query=q, meta=search_meta)
    except Exception:
        pass

    return jsonify({
        "query": q,
        "corrected_query": corrected if has_corr else None,
        "budget_myr": budget_myr,
        "results": results,
        "meta": {
            "total_indexed": len(PRODUCTS),
            "generated_at": datetime.now().isoformat(),
            "tolerance": BUDGET_TOLERANCE,
            "applied_correction": bool(has_corr),
            "fallback": fallback,
            "personalized": not opt_out,
            "embedding_mode": embedding_used,
            "simple_query_embedding": bool(embed_simple_used),
            "price_filter": price_range_to_list(price_range),
            "result_categories": top_categories,
            "result_subcategories": top_subcategories,
            "average_price": search_meta.get("avg_price"),
        }
    })



@app.route("/user/categories", methods=["GET"])
def user_categories():
    load_categories()
    user_id = request.headers.get("X-User-Id") or request.args.get("user_id") or "anon"
    since_secs = safe_int(request.args.get("since_secs"), 60 * 60 * 24 * 90)
    limit = safe_int(request.args.get("limit"), 500)

    events = get_recent_events(user_id, since_secs=since_secs, limit=limit)
    now = int(time.time())
    cat_counter = Counter()
    sub_counter = Counter()
    price_samples: list[tuple[float, float]] = []
    recent_queries: list[dict[str, object]] = []

    for ts, etype, query, product_id, meta_json in events:
        weight = decay_weight(now - ts)
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            meta = {}

        if etype == "click" and product_id:
            prod = get_product_by_id(product_id)
            if prod:
                cat_val = prod.get("main_category") or meta.get("main_category")
                if cat_val:
                    cat_counter[cat_val] += 2.0 * weight
                sub_val = prod.get("sub_category") or meta.get("sub_category")
                if sub_val:
                    sub_counter[sub_val] += 1.5 * weight
                price_val = prod.get("price_myr")
                if isinstance(price_val, (int, float)):
                    price_samples.append((float(price_val), 2.0 * weight))
        if etype == "search":
            if query:
                recent_queries.append({"query": query, "ts": ts})
                for cid in infer_category_ids_from_query(query):
                    name = CATEGORY_BY_ID.get(cid)
                    if name:
                        cat_counter[name] += 1.0 * weight
            categories_meta = meta.get("categories") or meta.get("category_names")
            if isinstance(categories_meta, list):
                for name in categories_meta:
                    if isinstance(name, str):
                        cat_counter[name] += 0.8 * weight
            sub_meta = meta.get("subcategories")
            if isinstance(sub_meta, list):
                for name in sub_meta:
                    if isinstance(name, str):
                        sub_counter[name] += 0.6 * weight
            price_meta = meta.get("applied_price_filter") or meta.get("price_range")
            if isinstance(price_meta, (list, tuple)) and len(price_meta) == 2:
                try:
                    values = [float(x) for x in price_meta if x is not None]
                except Exception:
                    values = []
                if values:
                    avg_val = sum(values) / len(values)
                    price_samples.append((avg_val, 0.6 * weight))

        meta_cat = meta.get("main_category")
        if isinstance(meta_cat, str):
            cat_counter[meta_cat] += 0.5 * weight
        meta_sub = meta.get("sub_category")
        if isinstance(meta_sub, str):
            sub_counter[meta_sub] += 0.5 * weight
        meta_price = meta.get("price_myr")
        if isinstance(meta_price, (int, float)):
            price_samples.append((float(meta_price), 0.5 * weight))

    if not cat_counter and GLOBAL_CATEGORY_COUNTS:
        for name, score in GLOBAL_CATEGORY_COUNTS.most_common(12):
            cat_counter[name] += float(score)
    if not sub_counter and GLOBAL_SUBCATEGORY_COUNTS:
        for name, score in GLOBAL_SUBCATEGORY_COUNTS.most_common(12):
            sub_counter[name] += float(score)

    def summarize(counter: Counter, limit: int = 12):
        return [
            {"label": name, "score": round(value, 3)}
            for name, value in counter.most_common(limit)
        ]

    price_summary = None
    if price_samples:
        values = [p for p, _ in price_samples]
        weights = [w for _, w in price_samples]
        min_price = min(values)
        max_price = max(values)
        weight_total = sum(weights) or 1.0
        avg_price = sum(p * w for p, w in price_samples) / weight_total
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2:
            median = sorted_vals[mid]
        else:
            median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        price_summary = {
            "min": round(min_price, 2),
            "max": round(max_price, 2),
            "average": round(avg_price, 2),
            "median": round(median, 2),
            "suggested_filter": [
                round(min_price, 2),
                round(max_price, 2),
            ],
        }

    catalog_categories = [
        {
            "label": name,
            "id": CATEGORY_NAME_TO_ID.get(name.lower()) if isinstance(name, str) else None,
        }
        for name, _ in GLOBAL_CATEGORY_COUNTS.most_common(20)
    ] if GLOBAL_CATEGORY_COUNTS else []

    response = {
        "user_id": user_id,
        "top_categories": summarize(cat_counter),
        "top_subcategories": summarize(sub_counter),
        "price_summary": price_summary,
        "recent_queries": recent_queries[:10],
        "catalog_categories": catalog_categories,
        "generated_at": datetime.now().isoformat(),
        "history_events": len(events),
    }
    return jsonify(response)

@app.route("/reload", methods=["POST"])
def reload_index():
    try:
        ensure_index()
        return jsonify({"message":"Index rebuilt", "total": len(PRODUCTS)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status":"ok", "indexed": len(PRODUCTS),
        "index_ready": INDEX is not None, "model": EMBED_MODEL_ID, "region": BEDROCK_REGION
    })

@app.route("/")
def home():
    return jsonify({"message": "Vector search API is running. Use GET/POST /search?query=... or POST /reload"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
