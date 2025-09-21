import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_URL = "/search";
const EVENTS_URL = "/event";
const ALLOW_OFFLINE = false;

const PRICE_MAX_DEFAULT = 1_000_000;

interface Product {
  id: string;
  title: string;
  price: number;
  currency: string;
  image?: string;
  rating?: number;
  reviews?: number;
  url?: string;
  badges?: string[];
  category?: string;
  brand?: string;
  inStock?: boolean;
}

interface FlaskItem {
  id: string;
  name: string;
  price_myr: number | null;
  image?: string;
  link?: string;
  ratings?: number;
  no_of_ratings?: number;
  main_category?: string;
  sub_category?: string;
}

interface FlaskResponse {
  query: string;
  corrected_query?: string | null;
  budget_myr?: number | null;
  results: FlaskItem[];
  meta: {
    fallback: boolean;
    total_indexed: number;
    generated_at: string;
    applied_correction: boolean;
    tolerance: number;
    personalized?: boolean;
    embedding_mode?: string;
    simple_query_embedding?: boolean;
    price_filter?: [number | null, number | null] | null;
    price_range?: [number, number] | null;
    result_categories?: string[];
    result_subcategories?: string[];
    average_price?: number | null;
  };
}

interface CategoryScore {
  label: string;
  score: number;
}

interface UserCategoryInsights {
  user_id: string;
  top_categories: CategoryScore[];
  top_subcategories: CategoryScore[];
  price_summary?: {
    min: number;
    max: number;
    average: number;
    median: number;
    suggested_filter?: [number, number];
  } | null;
  recent_queries: { query: string; ts: number }[];
  catalog_categories: { label: string; id?: string | null }[];
  generated_at: string;
  history_events: number;
}

type SortKey = "relevance" | "price-asc" | "price-desc" | "rating";

interface FiltersState {
  category: Set<string>;
  brand: Set<string>;
  stock: "any" | "in" | "out";
  priceRange?: [number, number];
}

const currencyFmt = (value: number, currency: string) => {
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency }).format(value);
  } catch {
    return `${currency} ${value.toFixed(2)}`;
  }
};

function classNames(...xs: Array<string | false | undefined>) {
  return xs.filter(Boolean).join(" ");
}

function debounce<T extends (...args: any[]) => void>(fn: T, ms: number) {
  let t: any;
  return (...args: Parameters<T>) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function highlight(text: string, query: string) {
  if (!query) return text;
  try {
    const re = new RegExp(`(${query.replace(/[-/\\\\^$*+?.()|[\\]{}]/g, "\\\\$&")})`, "ig");
    return (
      <span>
        {text.split(re).map((part, i) =>
          re.test(part) ? (
            <mark key={i} className="rounded bg-yellow-200 px-0.5">
              {part}
            </mark>
          ) : (
            <span key={i}>{part}</span>
          )
        )}
      </span>
    );
  } catch {
    return text;
  }
}

function sortResults(items: Product[], sort: SortKey) {
  const arr = [...items];
  switch (sort) {
    case "price-asc":
      arr.sort((a, b) => (a.price ?? 0) - (b.price ?? 0));
      break;
    case "price-desc":
      arr.sort((a, b) => (b.price ?? 0) - (a.price ?? 0));
      break;
    case "rating":
      arr.sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0));
      break;
    default:
      break;
  }
  return arr;
}

const LS_RECENT_QUERIES = "search-recent";
const LS_RECENT_TAXONOMY = "search-recent-taxonomy";
const LS_USER_ID = "uid";

function ensureUserId(): string {
  let uid = localStorage.getItem(LS_USER_ID);
  if (!uid) {
    uid = crypto.randomUUID();
    localStorage.setItem(LS_USER_ID, uid);
  }
  return uid;
}

export default function App() {
  const USER_ID = ensureUserId();

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Product[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FiltersState>({
    category: new Set(),
    brand: new Set(),
    stock: "any",
    priceRange: undefined,
  });
  const [sort, setSort] = useState<SortKey>("relevance");

  const [recent, setRecent] = useState<string[]>(() => {
    try {
      const raw = localStorage.getItem(LS_RECENT_QUERIES);
      const parsed = raw ? (JSON.parse(raw) as unknown) : [];
      return Array.isArray(parsed) ? (parsed as string[]) : [];
    } catch {
      return [];
    }
  });

  const [recentProducts, setRecentProducts] = useState<Product[]>([]);
  const [recentLoading, setRecentLoading] = useState(false);

  const [recentTaxonomy, setRecentTaxonomy] = useState<{ cat: Record<string, number>; sub: Record<string, number> }>(() => {
    try {
      const raw = localStorage.getItem(LS_RECENT_TAXONOMY);
      if (!raw) return { cat: {}, sub: {} };
      const parsed = JSON.parse(raw) as { cat?: Record<string, number>; sub?: Record<string, number> };
      return { cat: parsed.cat ?? {}, sub: parsed.sub ?? {} };
    } catch {
      return { cat: {}, sub: {} };
    }
  });

  const [activeIndex, setActiveIndex] = useState(-1);
  const [corrected, setCorrected] = useState<string | null>(null);
  const [budgetMYR, setBudgetMYR] = useState<number | null>(null);
  const [insights, setInsights] = useState<UserCategoryInsights | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [averagePrice, setAveragePrice] = useState<number | null>(null);
  const [suggestedPrice, setSuggestedPrice] = useState<[number, number] | null>(null);
  const [serverPriceFilter, setServerPriceFilter] = useState<[number | null, number | null] | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  const categories = useMemo(
    () => Array.from(new Set(results.map((p) => p.category).filter(Boolean))) as string[],
    [results]
  );
  const brands = useMemo(
    () => Array.from(new Set(results.map((p) => p.brand).filter(Boolean))) as string[],
    [results]
  );

  const runSearch = async (q: string, nextFilters = filters, nextSort = sort) => {
    setLoading(true);
    setError(null);
    setCorrected(null);
    setBudgetMYR(null);
    setAveragePrice(null);
    setServerPriceFilter(null);
    if (!nextFilters.priceRange) {
      setSuggestedPrice(null);
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const params = new URLSearchParams({ query: q });
      if (nextFilters.stock !== "any") params.set("stock", nextFilters.stock);
      if (nextFilters.category.size) params.set("category", Array.from(nextFilters.category).join(","));
      if (nextFilters.brand.size) params.set("brand", Array.from(nextFilters.brand).join(","));
      if (nextFilters.priceRange) {
        const [min, max] = nextFilters.priceRange;
        const hiPart = max >= PRICE_MAX_DEFAULT ? '' : String(max);
        const value = hiPart ? `${min}-${hiPart}` : `${min}-`;
        params.set('price', value);
      }
      if (nextSort !== "relevance") params.set("sort", nextSort);

      const res = await fetch(`${API_URL}?${params.toString()}`, {
        signal: controller.signal,
        headers: { "X-User-Id": USER_ID },
      });
      const data: FlaskResponse = await res.json();
      if (!res.ok) {
        throw new Error((data as any)?.error ?? `HTTP ${res.status}`);
      }

      if (data.meta?.fallback && !ALLOW_OFFLINE) {
        setResults([]);
        setSuggestions([]);
        setError("Backend is in fallback mode. Dataset-only view is enabled.");
        return;
      }

      const mapped: Product[] = (data.results ?? []).map((r) => ({
        id: r.id,
        title: r.name,
        price: typeof r.price_myr === "number" ? r.price_myr : 0,
        currency: "MYR",
        image: r.image,
        rating: r.ratings ?? undefined,
        reviews: r.no_of_ratings ?? undefined,
        url: r.link,
        category: r.main_category || undefined,
        brand: r.sub_category || undefined,
        inStock: true,
      }));

      let pool = mapped;
      if (nextFilters.category.size) {
        pool = pool.filter((p) => p.category && nextFilters.category.has(p.category));
      }
      if (nextFilters.brand.size) {
        pool = pool.filter((p) => p.brand && nextFilters.brand.has(p.brand));
      }
      if (nextFilters.stock !== "any") {
        pool = pool.filter((p) => (nextFilters.stock === "in" ? p.inStock : !p.inStock));
      }
      if (nextFilters.priceRange) {
        const [min, max] = nextFilters.priceRange;
        pool = pool.filter((p) => p.price >= min && p.price <= max);
      }
      pool = sortResults(pool, nextSort);

      const ql = q.trim().toLowerCase();
      const sugg: string[] =
        ql && pool.length
          ? Array.from(
              new Set(
                pool.flatMap((p) =>
                  (p.title || "")
                    .toLowerCase()
                    .split(/\s+/)
                    .filter((w) => w.startsWith(ql) && w.length > ql.length)
                )
              )
            ).slice(0, 6)
          : [];

      setResults(pool);
      setSuggestions(sugg ?? []);
      setCorrected(data.corrected_query ?? null);
      setBudgetMYR(typeof data.budget_myr === "number" ? data.budget_myr : null);

      setAveragePrice(typeof data.meta?.average_price === 'number' ? data.meta!.average_price : null);

      if (Array.isArray(data.meta?.price_filter) && data.meta!.price_filter.length === 2) {
        const [lo, hi] = data.meta!.price_filter;
        const normalized: [number | null, number | null] = [
          lo == null || Number.isNaN(Number(lo)) ? null : Number(lo),
          hi == null || Number.isNaN(Number(hi)) ? null : Number(hi),
        ];
        setServerPriceFilter(normalized);
      } else {
        setServerPriceFilter(null);
      }

      if (!nextFilters.priceRange && Array.isArray(data.meta?.price_range) && data.meta!.price_range.length === 2) {
        const [lo, hi] = data.meta!.price_range;
        if (typeof lo === 'number' && typeof hi === 'number') {
          setSuggestedPrice([lo, hi]);
        }
      }

      if (pool.length) {
        const nextCounts = { cat: { ...recentTaxonomy.cat }, sub: { ...recentTaxonomy.sub } };
        for (const p of pool) {
          if (p.category) nextCounts.cat[p.category] = (nextCounts.cat[p.category] ?? 0) + 1;
          if (p.brand) nextCounts.sub[p.brand] = (nextCounts.sub[p.brand] ?? 0) + 1;
        }
        const metaCats = Array.isArray(data.meta?.result_categories) ? data.meta!.result_categories : [];
        metaCats.forEach((c, idx) => {
          if (!c) return;
          nextCounts.cat[c] = (nextCounts.cat[c] ?? 0) + 2 + Math.max(0, metaCats.length - idx);
        });
        const metaSubs = Array.isArray(data.meta?.result_subcategories) ? data.meta!.result_subcategories : [];
        metaSubs.forEach((s, idx) => {
          if (!s) return;
          nextCounts.sub[s] = (nextCounts.sub[s] ?? 0) + 2 + Math.max(0, metaSubs.length - idx);
        });
        setRecentTaxonomy(nextCounts);
        try {
          localStorage.setItem(LS_RECENT_TAXONOMY, JSON.stringify(nextCounts));
        } catch {}
      } else {
        setSuggestedPrice(null);
      }
      fetchInsights();

    } catch (e: any) {
      if (e?.name === "AbortError") return;
      setResults([]);
      setSuggestions([]);
      setError("Backend unreachable. Dataset-only mode disables offline demo.");
    } finally {
      setLoading(false);
    }
  };

  const debouncedSearch = useMemo(() => debounce(runSearch, 250), [filters, sort]);
  const fetchInsights = useCallback(async () => {
    setInsightsLoading(true);
    try {
      const res = await fetch('/user/categories', {
        headers: { 'X-User-Id': USER_ID },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: UserCategoryInsights = await res.json();
      setInsights(data);
      if (data.price_summary?.suggested_filter && !filters.priceRange) {
        const [lo, hi] = data.price_summary.suggested_filter;
        if (typeof lo === 'number' && typeof hi === 'number') {
          setSuggestedPrice([lo, hi]);
        }
      }
    } catch (err) {
      console.warn('Failed to load category insights', err);
    } finally {
      setInsightsLoading(false);
    }
  }, [USER_ID, filters.priceRange]);

  const loadRecentProducts = async () => {
    if (!recent.length) return;
    setRecentLoading(true);
    try {
      const queries = recent.slice(0, 3);
      const fetchedArrays = await Promise.all(
        queries.map(async (q) => {
          try {
            const res = await fetch(`${API_URL}?${new URLSearchParams({ query: q }).toString()}`, {
              headers: { "X-User-Id": USER_ID },
            });
            const data: FlaskResponse = await res.json();
            if (!res.ok) return [] as Product[];
            if (data.meta?.fallback && !ALLOW_OFFLINE) return [] as Product[];
            const mapped: Product[] = (data.results ?? []).map((r) => ({
              id: r.id,
              title: r.name,
              price: typeof r.price_myr === "number" ? r.price_myr : 0,
              currency: "MYR",
              image: r.image,
              rating: r.ratings ?? undefined,
              reviews: r.no_of_ratings ?? undefined,
              url: r.link,
              category: r.main_category || undefined,
              brand: r.sub_category || undefined,
              inStock: true,
            }));
            return mapped;
          } catch {
            return [] as Product[];
          }
        })
      );
      const mergedMap = new Map<string, Product>();
      for (const arr of fetchedArrays) {
        for (const p of arr) {
          if (!mergedMap.has(p.id)) mergedMap.set(p.id, p);
        }
      }
      setRecentProducts(Array.from(mergedMap.values()).slice(0, 24));
    } finally {
      setRecentLoading(false);
    }
  };

  useEffect(() => {
    if (recent.length > 0) {
      loadRecentProducts();
    } else {
      runSearch("");
    }
  }, []);

  useEffect(() => {
    fetchInsights();
  }, [fetchInsights]);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!boxRef.current) return;
      if (!boxRef.current.contains(e.target as Node)) setActiveIndex(-1);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(LS_RECENT_QUERIES, JSON.stringify(recent.slice(0, 10)));
    } catch {}
  }, [recent]);

  const onSubmit = (q: string) => {
    if (!q.trim()) return;
    setRecent((prev) => [q, ...prev.filter((x) => x !== q)].slice(0, 10));
    runSearch(q);
  };
  const onChange = (q: string) => {
    setQuery(q);
    debouncedSearch(q);
  };

  const clearAll = () => {
    const f2: FiltersState = { category: new Set(), brand: new Set(), stock: "any", priceRange: undefined };
    setFilters(f2);
    setSort("relevance");
    runSearch(query, f2, "relevance");
  };

  const activeItemUrl = () => {
    const list = suggestions.length ? suggestions : results.map((r) => r.url || "#");
    if (activeIndex < 0 || activeIndex >= list.length) return null;
    if (suggestions.length) return `?q=${encodeURIComponent(suggestions[activeIndex])}`;
    return list[activeIndex] || null;
  };

  const topCats = useMemo(() => {
    const scores = new Map<string, number>();
    Object.entries(recentTaxonomy.cat).forEach(([label, count]) => {
      scores.set(label, (scores.get(label) ?? 0) + count);
    });
    if (insights?.top_categories) {
      insights.top_categories.forEach((item, idx) => {
        if (!item?.label) return;
        const weight = Number.isFinite(item.score) ? item.score : 0;
        const bonus = Math.max(0, (insights.top_categories?.length || 0) - idx);
        scores.set(item.label, Math.max(scores.get(item.label) ?? 0, weight + bonus));
      });
    }
    return Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([label]) => label)
      .slice(0, 12);
  }, [recentTaxonomy, insights]);

  const topSubs = useMemo(() => {
    const scores = new Map<string, number>();
    Object.entries(recentTaxonomy.sub).forEach(([label, count]) => {
      scores.set(label, (scores.get(label) ?? 0) + count);
    });
    if (insights?.top_subcategories) {
      insights.top_subcategories.forEach((item, idx) => {
        if (!item?.label) return;
        const weight = Number.isFinite(item.score) ? item.score : 0;
        const bonus = Math.max(0, (insights.top_subcategories?.length || 0) - idx);
        scores.set(item.label, Math.max(scores.get(item.label) ?? 0, weight + bonus));
      });
    }
    return Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([label]) => label)
      .slice(0, 12);
  }, [recentTaxonomy, insights]);

  const onOpen = (p: Product) => {
    fetch(EVENTS_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: USER_ID, type: "click", product_id: p.id }),
    })
      .catch(() => {})
      .finally(() => {
        if (p.url) window.open(p.url, "_blank");
      });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-slate-900 antialiased">
      <header className="sticky top-0 z-30 backdrop-blur supports-[backdrop-filter]:bg-white/70 bg-white/60 border-b">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="h-9 w-9 rounded-2xl bg-slate-900 text-white grid place-items-center font-bold">AI</div>
            <h1 className="text-lg font-semibold tracking-tight">Website Search Assistant</h1>
          </div>
          <div className="ml-auto text-sm text-slate-500 hidden md:block">
            Instant Results • Typing to Search • ↑/↓ then Enter
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        {/* Search Box */}
        <div ref={boxRef} className="relative">
          <div className="group flex items-center gap-2 rounded-2xl border bg-white px-4 py-3 shadow-sm ring-1 ring-transparent focus-within:ring-slate-900/10">
            <svg className="h-5 w-5 opacity-60" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="7" />
              <path d="m20 20-3.5-3.5" />
            </svg>
            <input
              value={query}
              onChange={(e) => onChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setActiveIndex((i) => Math.min(i + 1, Math.max(suggestions.length - 1, results.length - 1)));
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setActiveIndex((i) => Math.max(i - 1, -1));
                } else if (e.key === "Enter") {
                  const url = activeItemUrl();
                  if (url) window.location.href = url;
                  else onSubmit(query);
                }
              }}
              placeholder="Search products, brands, categories…"
              className="flex-1 bg-transparent outline-none placeholder:text-slate-400"
            />
            {query && (
              <button
                onClick={() => {
                  setQuery("");
                  setSuggestions([]);
                  runSearch("");
                }}
                className="rounded-xl px-2 py-1 text-xs text-slate-600 hover:bg-slate-100"
              >
                Clear
              </button>
            )}
            <kbd className="hidden md:block rounded-lg border bg-slate-50 px-1.5 py-0.5 text-[11px] text-slate-500">/</kbd>
          </div>

          {/* Suggestion + Recent 面板 ——— 改为非绝对定位，避免遮挡下方标签 */}
          {(suggestions.length > 0 || recent.length > 0 || topCats.length > 0 || topSubs.length > 0) && (
            <div className="mt-2 rounded-2xl border bg-white shadow-xl">
              {suggestions.length > 0 && (
                <div className="p-2">
                  <div className="px-2 pb-1 text-xs font-medium uppercase text-slate-500">Suggestions</div>
                  <div className="max-h-64 overflow-auto">
                    {suggestions.map((s, idx) => (
                      <button
                        key={s + idx}
                        onMouseEnter={() => setActiveIndex(idx)}
                        onMouseLeave={() => setActiveIndex(-1)}
                        onClick={() => {
                          setQuery(s);
                          onSubmit(s);
                          setSuggestions([]);
                        }}
                        className={classNames(
                          "flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left hover:bg-slate-50",
                          activeIndex === idx && "bg-slate-100"
                        )}
                      >
                        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M4 7h16M4 12h16M4 17h16" />
                        </svg>
                        <span>{highlight(s, query)}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {(recent.length > 0 || topCats.length > 0 || topSubs.length > 0) && (
                <div className="border-t p-2">
                  <div className="px-2 pb-1 text-xs font-medium uppercase text-slate-500">Recent</div>
                  {/* 查询历史 chips */}
                  {recent.length > 0 && (
                    <div className="flex flex-wrap gap-2 px-2 pb-2">
                      {recent.map((r) => (
                        <button
                          key={r}
                          onClick={() => {
                            setQuery(r);
                            onSubmit(r);
                            setSuggestions([]);
                          }}
                          className="rounded-full border px-3 py-1 text-sm hover:bg-slate-50"
                        >
                          {r}
                        </button>
                      ))}
                    </div>
                  )}
                  {/* 动态分类 chips：来自真实 datasets 字段 */}
                  {(topCats.length > 0 || topSubs.length > 0) && (
                    <div className="flex flex-col gap-2 px-2 pb-2">
                      {topCats.length > 0 && (
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-xs text-slate-500">Categories</span>
                          {topCats.map((c) => (
                            <button
                              key={c}
                              onClick={() => {
                                const next = new Set(filters.category);
                                next.has(c) ? next.delete(c) : next.add(c);
                                const f2 = { ...filters, category: next };
                                setFilters(f2);
                                runSearch(query, f2);
                              }}
                              className={classNames(
                                "rounded-full border px-3 py-1 text-sm",
                                filters.category.has(c) ? "border-slate-900 bg-slate-900 text-white" : "hover:bg-slate-50"
                              )}
                            >
                              {c}
                            </button>
                          ))}
                        </div>
                      )}
                      {topSubs.length > 0 && (
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-xs text-slate-500">Subcategories</span>
                          {topSubs.map((b) => (
                            <button
                              key={b}
                              onClick={() => {
                                const next = new Set(filters.brand);
                                next.has(b) ? next.delete(b) : next.add(b);
                                const f2 = { ...filters, brand: next };
                                setFilters(f2);
                                runSearch(query, f2);
                              }}
                              className={classNames(
                                "rounded-full border px-3 py-1 text-sm",
                                filters.brand.has(b) ? "border-slate-900 bg-slate-900 text-white" : "hover:bg-slate-50"
                              )}
                            >
                              {b}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* 最近搜索过的产品（仅在未输入查询且有历史时显示） */}
        {results.length > 0 && (
          <div className="mt-4 flex flex-col gap-3 rounded-2xl border bg-white px-4 py-3 shadow-sm">
            <div className="flex flex-wrap items-center gap-3">
              {topCats.length > 0 && (
                <Facet
                  label="Categories"
                  options={(topCats.length ? topCats : categories).slice(0, 6)}
                  selected={filters.category}
                  onToggle={(value) => {
                    const next = new Set(filters.category);
                    next.has(value) ? next.delete(value) : next.add(value);
                    const f2: FiltersState = { ...filters, category: next };
                    setFilters(f2);
                    runSearch(query, f2, sort);
                  }}
                />
              )}
              {topSubs.length > 0 && (
                <Facet
                  label="Subcategories"
                  options={(topSubs.length ? topSubs : brands).slice(0, 6)}
                  selected={filters.brand}
                  onToggle={(value) => {
                    const next = new Set(filters.brand);
                    next.has(value) ? next.delete(value) : next.add(value);
                    const f2: FiltersState = { ...filters, brand: next };
                    setFilters(f2);
                    runSearch(query, f2, sort);
                  }}
                />
              )}
              <Segmented
                label="Sort"
                value={sort}
                options={[
                  { value: 'relevance' as SortKey, label: 'Relevance' },
                  { value: 'price-asc' as SortKey, label: 'Price Asc' },
                  { value: 'price-desc' as SortKey, label: 'Price Desc' },
                  { value: 'rating' as SortKey, label: 'Rating' },
                ]}
                onChange={(next) => {
                  setSort(next);
                  runSearch(query, filters, next);
                }}
              />
              {insightsLoading && (
                <span className="text-xs text-slate-400">Updating preferences…</span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <PriceRange
                value={filters.priceRange}
                suggestion={suggestedPrice}
                onChange={(range) => {
                  const f2: FiltersState = { ...filters, priceRange: range ? range : undefined };
                  setFilters(f2);
                  runSearch(query, f2, sort);
                }}
              />
              {serverPriceFilter && (
                <span className="text-xs text-slate-500">
                  Applied filter: {serverPriceFilter[0] != null ? currencyFmt(serverPriceFilter[0], 'MYR') : 'Any'}
                  {' - '}
                  {serverPriceFilter[1] != null ? currencyFmt(serverPriceFilter[1], 'MYR') : 'Any'}
                </span>
              )}
            </div>
          </div>
        )}

        {query.trim() === "" && recent.length > 0 && (
          <section className="mt-6">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-base font-semibold">Based on your recent searches</h2>
              <div className="text-sm text-slate-500">{recent.slice(0,3).join(" · ")}</div>
            </div>
            {recentLoading ? (
              <div className="text-sm text-slate-500"><Spinner /> Loading your recent items…</div>
            ) : recentProducts.length > 0 ? (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {recentProducts.map((p) => (
                  <article key={p.id} className="group overflow-hidden rounded-2xl border bg-white shadow-sm transition hover:shadow-md">
                    <div className="relative aspect-[4/3] overflow-hidden">
                      {p.image ? (
                        <img src={p.image} alt={p.title} className="h-full w-full object-cover transition group-hover:scale-[1.03]" loading="lazy" />
                      ) : (
                        <div className="grid h-full w-full place-items-center bg-slate-100 text-slate-400">No Image</div>
                      )}
                    </div>
                    <div className="p-4">
                      <h3 className="line-clamp-2 text-sm font-medium text-slate-900">{p.title}</h3>
                      <div className="mt-2 flex items-center justify-between">
                        <div className="text-base font-semibold">{currencyFmt(p.price ?? 0, p.currency)}</div>
                        <div className="flex items-center gap-1 text-sm text-slate-500">
                          <StarRating value={p.rating ?? 0} />
                          <span>({p.reviews ?? 0})</span>
                        </div>
                      </div>
                      <div className="mt-2 flex items-center gap-2 text-xs text-slate-500">
                        {p.brand && <span className="rounded-full border px-2 py-0.5">{p.brand}</span>}
                        {p.category && <span className="rounded-full border px-2 py-0.5">{p.category}</span>}
                      </div>
                      <div className="mt-3">
                        <button onClick={() => onOpen(p)} className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-3 py-2 text-sm text-white hover:bg-slate-800">
                          View
                          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M7 17 17 7M7 7h10v10" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="rounded-2xl border bg-white p-6 text-sm text-slate-500">No items from your recent searches yet.</div>
            )}
          </section>
        )}

        {/* Results */}
        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {results.map((p) => (
            <article key={p.id} className="group overflow-hidden rounded-2xl border bg-white shadow-sm transition hover:shadow-md">
              <div className="relative aspect-[4/3] overflow-hidden">
                {p.image ? (
                  <img
                    src={p.image}
                    alt={p.title}
                    className="h-full w-full object-cover transition group-hover:scale-[1.03]"
                    loading="lazy"
                  />
                ) : (
                  <div className="grid h-full w-full place-items-center bg-slate-100 text-slate-400">No Image</div>
                )}
                {p.badges?.length ? (
                  <div className="pointer-events-none absolute left-2 top-2 flex gap-1">
                    {p.badges.map((b) => (
                      <span key={b} className="rounded-full bg-slate-900/90 px-2 py-0.5 text-xs text-white">
                        {b}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
              <div className="p-4">
                <h3 className="line-clamp-2 text-sm font-medium text-slate-900">{highlight(p.title, query)}</h3>
                <div className="mt-2 flex items-center justify-between">
                  <div className="text-base font-semibold">{currencyFmt(p.price ?? 0, p.currency)}</div>
                  <div className="flex items-center gap-1 text-sm text-slate-500">
                    <StarRating value={p.rating ?? 0} />
                    <span>({p.reviews ?? 0})</span>
                  </div>
                </div>
                <div className="mt-2 flex items-center gap-2 text-xs text-slate-500">
                  {p.brand && <span className="rounded-full border px-2 py-0.5">{p.brand}</span>}
                  {p.category && <span className="rounded-full border px-2 py-0.5">{p.category}</span>}
                  <span className={classNames("rounded-full px-2 py-0.5", p.inStock ? "border border-emerald-300 text-emerald-700" : "border border-rose-300 text-rose-700")}>
                    {p.inStock ? "In stock" : "Out of stock"}
                  </span>
                </div>
                <div className="mt-3 flex items-center gap-2">
                  <button
                    onClick={() => onOpen(p)}
                    className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-3 py-2 text-sm text-white hover:bg-slate-800"
                  >
                    View
                    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M7 17 17 7M7 7h10v10" />
                    </svg>
                  </button>
                  <button className="rounded-xl border px-3 py-2 text-sm hover:bg-slate-50">Add to cart</button>
                </div>
              </div>
            </article>
          ))}
        </div>

        {/* Status */}
        <div className="mt-3 text-sm text-slate-500">
          {loading ? (
            <span className="inline-flex items-center gap-2">
              <Spinner /> Searching…
            </span>
          ) : (
            <span>
              Showing <strong>{results.length}</strong> result{results.length === 1 ? "" : "s"}
              {query ? (
                <> for <strong>“{query}”</strong></>
              ) : (
                <> (popular)</>
              )}
            </span>
          )}
          {corrected && corrected !== query && (
            <span className="ml-3">Did you mean “<em>{corrected}</em>”?</span>
          )}
          {typeof budgetMYR === "number" && <span className="ml-3">Budget ≈ MYR {budgetMYR.toFixed(2)}</span>}
          {averagePrice != null && <span className="ml-3">Avg price {currencyFmt(averagePrice, "MYR")}</span>}
          {error && <span className="ml-3 text-amber-600">{error}</span>}
        </div>

        {/* Empty state */}
        {!loading && results.length === 0 && (
          <div className="mt-14 grid place-items-center rounded-3xl border bg-white p-10 text-center">
            <div className="mx-auto max-w-lg">
              <div className="mx-auto mb-2 grid h-12 w-12 place-items-center rounded-2xl bg-slate-900 text-white">🤖</div>
              <h3 className="text-lg font-semibold">No dataset results</h3>
              <p className="mt-1 text-slate-500">Only products from the backend datasets are shown. Try another keyword or adjust filters.</p>
              <div className="mt-4 flex justify-center gap-2">
                <button onClick={() => { setQuery(""); runSearch(""); }} className="rounded-xl border px-3 py-2 text-sm hover:bg-slate-50">Reset search</button>
                <button onClick={clearAll} className="rounded-xl border px-3 py-2 text-sm hover:bg-slate-50">Clear filters</button>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="mx-auto max-w-6xl px-4 pb-10 pt-6 text-center text-sm text-slate-500">
        <p>
          Tip: Press <kbd className="rounded border bg-slate-50 px-1">/</kbd> to focus the search box. Use <kbd className="rounded border bg-slate-50 px-1">↑</kbd>/<kbd className="rounded border bg-slate-50 px-1">↓</kbd> for quick navigation.
        </p>
      </footer>
    </div>
  );
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity=".25" strokeWidth="4" />
      <path d="M22 12a10 10 0 0 1-10 10" stroke="currentColor" strokeWidth="4" />
    </svg>
  );
}

function StarRating({ value = 0 }: { value?: number }) {
  const stars = Math.max(0, Math.min(5, Math.round(value)));
  return (
    <div className="flex items-center">
      {Array.from({ length: 5 }).map((_, i) => (
        <svg
          key={i}
          className={classNames("h-4 w-4", i < stars ? "fill-yellow-400 stroke-yellow-400" : "fill-none stroke-slate-300")}
          viewBox="0 0 24 24"
        >
          <path d="m12 17.27 5.18 3.04-1.64-5.81L20 9.24l-5.92-.51L12 3 9.92 8.73 4 9.24l4.46 5.26-1.64 5.81L12 17.27Z" />
        </svg>
      ))}
    </div>
  );
}

function Facet({ label, options, selected, onToggle }: { label: string; options: string[]; selected: Set<string>; onToggle: (value: string) => void; }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-slate-500">{label}</span>
      <div className="flex flex-wrap gap-2">
        {options.map((opt) => (
          <button
            key={opt}
            onClick={() => onToggle(opt)}
            className={classNames(
              "rounded-full border px-3 py-1 text-sm",
              selected.has(opt) ? "border-slate-900 bg-slate-900 text-white" : "hover:bg-slate-50"
            )}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}

function Segmented<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-slate-500">{label}</span>
      <div className="inline-flex overflow-hidden rounded-xl border">
        {options.map((o) => (
          <button
            key={String(o.value)}
            onClick={() => onChange(o.value)}
            className={classNames(
              "px-3 py-1 text-sm",
              o.value === value ? "bg-slate-900 text-white" : "bg-white hover:bg-slate-50"
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function PriceRange({ value, suggestion, onChange }: { value?: [number, number]; suggestion?: [number, number] | null; onChange: (range?: [number, number]) => void }) {
  const [min, setMin] = useState<string>('');
  const [max, setMax] = useState<string>('');

  useEffect(() => {
    if (value) {
      const [lo, hi] = value;
      setMin(lo != null ? String(lo) : '');
      setMax(hi != null && hi < PRICE_MAX_DEFAULT ? String(hi) : '');
    } else {
      setMin('');
      setMax('');
    }
  }, [value?.[0], value?.[1]]);

  const parsedMin = min ? Number(min) : undefined;
  const parsedMax = max ? Number(max) : undefined;
  const minInvalid = min.length > 0 && Number.isNaN(parsedMin);
  const maxInvalid = max.length > 0 && Number.isNaN(parsedMax);

  const applyRange = (lo?: number, hi?: number) => {
    if (lo == null && hi == null) {
      onChange(undefined);
      return;
    }
    const range: [number, number] = [lo ?? 0, hi ?? PRICE_MAX_DEFAULT];
    onChange(range);
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-sm text-slate-500">Price</span>
      <input
        type="number"
        value={min}
        onChange={(e) => setMin(e.target.value)}
        placeholder="Min"
        className="w-20 rounded-xl border bg-white px-2 py-1 text-sm"
      />
      <span className="text-slate-400">-</span>
      <input
        type="number"
        value={max}
        onChange={(e) => setMax(e.target.value)}
        placeholder="Max"
        className="w-20 rounded-xl border bg-white px-2 py-1 text-sm"
      />
      <button
        onClick={() => {
          if (minInvalid || maxInvalid) return;
          applyRange(parsedMin, parsedMax);
        }}
        className="rounded-xl border px-3 py-1 text-sm hover:bg-slate-50"
      >
        Apply
      </button>
      <button
        onClick={() => {
          setMin('');
          setMax('');
          onChange(undefined);
        }}
        className="rounded-xl border px-3 py-1 text-sm text-slate-500 hover:bg-slate-50"
      >
        Clear
      </button>
      {suggestion && (
        <button
          onClick={() => {
            setMin(String(suggestion[0]));
            setMax(String(suggestion[1]));
            applyRange(suggestion[0], suggestion[1]);
          }}
          className="rounded-xl border px-3 py-1 text-sm text-slate-600 hover:bg-slate-50"
        >
          Use {currencyFmt(suggestion[0], 'MYR')} - {currencyFmt(suggestion[1], 'MYR')}
        </button>
      )}
      {(minInvalid || maxInvalid) && (
        <span className="text-xs text-rose-500">Enter valid numbers</span>
      )}
    </div>
  );
}

