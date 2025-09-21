import math, time, numpy as np
from collections import Counter

HALF_LIFE_DAYS = 7
LAMBDA = math.log(2) / (HALF_LIFE_DAYS * 24 * 3600)

def decay_weight(delta_sec: int) -> float:
    return math.exp(-LAMBDA * delta_sec)

def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))

def budget_fit(budget, price):
    if budget is None or price is None: return 0.0
    if price <= 0: return 0.0
    sigma = 0.3 * budget
    return math.exp(-((price-budget)**2) / (2*(sigma**2 + 1e-9)))

def build_user_profile(user_id: str, get_prod_vec, get_query_vec, get_product_meta, get_recent_events_fn=None):
    """
    get_prod_vec(pid) -> np.ndarray
    get_query_vec(q)  -> np.ndarray
    get_product_meta(pid) -> dict(main_category, sub_category, brand, price_myr)
    get_recent_events_fn() -> [(ts, type, query, product_id, meta_json)]
    """
    if get_recent_events_fn is None:
        from events_store import get_recent_events as _gre
        events = _gre(user_id, since_secs=3600*24*90, limit=4000)
    else:
        events = get_recent_events_fn(user_id)

    now = int(time.time())
    vecs, ws = [], []
    cat_counter, sub_counter, brand_counter = Counter(), Counter(), Counter()
    budgets = []
    recent_clicks = set()

    for ts, etype, query, product_id, meta_json in events:
        w = decay_weight(now - ts)
        if etype == "click" and product_id:
            pv = get_prod_vec(product_id)
            if pv is not None:
                vecs.append(pv); ws.append(2.0*w)  # 点击权重更高
            pm = get_product_meta(product_id) or {}
            if pm.get("main_category"): cat_counter[pm["main_category"]] += w
            if pm.get("sub_category"):  sub_counter[pm["sub_category"]]  += w
            if pm.get("brand"):         brand_counter[pm["brand"]]       += w
            if pm.get("price_myr"):     budgets.append((pm["price_myr"], w))
            if (now - ts) <= 24*3600:
                recent_clicks.add(product_id)
        elif etype == "search" and query:
            qv = get_query_vec(query)
            if qv is not None:
                vecs.append(qv); ws.append(0.8*w)

    user_vec = None
    if vecs:
        W = np.array(ws, dtype="float32")
        M = np.vstack(vecs)
        user_vec = normalize((M * W[:,None]).sum(axis=0))

    def to_aff(cnt):
        tot = float(sum(cnt.values()) or 1.0)
        return {k: float(v/tot) for k,v in cnt.items()}

    profile = {
        "user_vec": user_vec,
        "cat_aff": to_aff(cat_counter),
        "sub_aff": to_aff(sub_counter),
        "brand_aff": to_aff(brand_counter),
        "budget": (sum(p*w for p,w in budgets)/ (sum(w for _,w in budgets) or 1.0)) if budgets else None,
        "recent_clicks": recent_clicks,
    }
    return profile

def affinity_score(aff_map, key):
    if not key: return 0.0
    return float(aff_map.get(key, 0.0))

def rerank(query_vec, user_profile, candidates, get_prod_vec, get_product_meta, weights=None, epsilon=0.02):
    W = {"w_q":0.55, "w_u":0.25, "w_c":0.06, "w_b":0.04, "w_price":0.06, "w_rc":0.04}
    if weights: W.update(weights)

    uv = user_profile.get("user_vec")
    cat_aff = user_profile.get("cat_aff", {})
    sub_aff = user_profile.get("sub_aff", {})
    brand_aff = user_profile.get("brand_aff", {})
    budget = user_profile.get("budget")
    rc = user_profile.get("recent_clicks", set())

    out = []
    for pid in candidates:
        pv = get_prod_vec(pid)
        if pv is None:
            out.append((pid, -1e9)); continue
        meta = get_product_meta(pid) or {}
        s = 0.0
        if query_vec is not None: s += W["w_q"] * cosine(query_vec, pv)
        if uv is not None:        s += W["w_u"] * cosine(uv, pv)
        s += W["w_c"] * affinity_score(cat_aff,  meta.get("main_category"))
        s += W["w_b"] * max(affinity_score(sub_aff, meta.get("sub_category")), affinity_score(brand_aff, meta.get("brand")))
        s += W["w_price"] * budget_fit(budget, meta.get("price_myr"))
        if pid in rc: s += W["w_rc"]
        s += float(np.random.uniform(0, epsilon))
        out.append((pid, s))
    out.sort(key=lambda x:x[1], reverse=True)
    return out
