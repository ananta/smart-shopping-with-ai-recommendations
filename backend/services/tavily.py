# -----------------------------------------------------------------------------
# Tavily API integration for product pricing and purchase links
# -----------------------------------------------------------------------------

# services/tavily.py  (or wherever your function lives)
import os, re, time, requests
from urllib.parse import urlparse, quote_plus
from typing import Dict, Any, List, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Tunables (env overridable) ---
CONNECT_TIMEOUT = float(os.getenv("PRICING_CONNECT_TIMEOUT", "1.5"))
READ_TIMEOUT_FAST = float(os.getenv("PRICING_READ_TIMEOUT_FAST", "3.5"))   # first try
READ_TIMEOUT_SLOW = float(os.getenv("PRICING_READ_TIMEOUT_SLOW", "7.5"))   # fallback
CACHE_TTL = int(os.getenv("PRICING_CACHE_TTL", "600"))                     # 10 min
MAX_RESULTS = int(os.getenv("PRICING_MAX_RESULTS", "8"))
FAST_MODE = os.getenv("PRICING_FAST_MODE", "1") == "1"                     # skip slow fallback

VENDOR_DOMAINS = [
    "amazon.com","bestbuy.com","walmart.com","target.com","newegg.com",
    "bhphotovideo.com","microcenter.com"
]

PRICE_RX = [
    r"\$[\d,]+(?:\.\d+)?",
    r"[\d,]+(?:\.\d+)?\s?(?:usd|dollars?)",
    r"€\s?[\d\.]+",
    r"£\s?[\d\.]+",
    r"(?:inr|₹)\s?[\d,]+(?:\.\d+)?",
]

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        return ""

def _vendor_from_domain(d: str) -> str:
    d = d.lower()
    if d.endswith("amazon.com"): return "amazon"
    if d.endswith("bestbuy.com"): return "bestbuy"
    if d.endswith("walmart.com"): return "walmart"
    if d.endswith("target.com"): return "target"
    if d.endswith("newegg.com"): return "newegg"
    if d.endswith("bhphotovideo.com"): return "bhphoto"
    if d.endswith("microcenter.com"): return "microcenter"
    return d or "unknown"

def _pdp_score(u: str) -> int:
    u = (u or "").lower()
    return int(any(x in u for x in ["/dp/","/gp/product/","/p/","/sku/","/product/"]))

def _parse_price(text: str) -> Optional[str]:
    for rx in PRICE_RX:
        m = re.search(rx, text or "", flags=re.IGNORECASE)
        if m: return m.group(0)
    return None
# ---- requests session with retry on HTTP 5xx/429 (timeouts retried manually) ----
_session = requests.Session()
_retry = Retry(total=2, backoff_factor=0.5,
               status_forcelist=(429,500,502,503,504),
               allowed_methods=frozenset(["POST"]))
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=10, pool_maxsize=20)
_session.mount("https://", _adapter); _session.mount("http://", _adapter)

# ---- tiny in-proc cache ----
_cache: Dict[Tuple[str, Tuple[str,...], int], Tuple[float, Dict[str, Any]]] = {}

def _cache_get(key):
    hit = _cache.get(key)
    if not hit: return None
    exp, data = hit
    if time.time() > exp:
        _cache.pop(key, None); return None
    return data

def _cache_set(key, data):
    _cache[key] = (time.time() + CACHE_TTL, data)

def _tavily_post(payload: Dict[str, Any], read_timeout: float) -> Dict[str, Any]:
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "content-type": "application/json"}
    resp = _session.post(url, headers=headers, json=payload, timeout=(CONNECT_TIMEOUT, read_timeout))
    resp.raise_for_status()
    return resp.json()

def _tavily_search(query: str, include_domains: Optional[List[str]], max_results: int) -> Dict[str, Any]:
    """Fast then slow attempt; both keep payload small to reduce latency."""
    if not TAVILY_API_KEY:
        raise RuntimeError("Tavily API key not configured")

    key = (query, tuple(include_domains or ()), max_results)
    cached = _cache_get(key)
    if cached: return cached

    base = {
        "query": query,
        "include_raw_content": False,
        "include_answer": False,   # smaller payload = faster
        "search_depth": "basic",   # faster than "advanced"
        "max_results": max_results,
    }
    if include_domains: base["include_domains"] = include_domains

    # Attempt 1: fast read timeout
    try:
        data = _tavily_post(base, READ_TIMEOUT_FAST)
        _cache_set(key, data); return data
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
        if FAST_MODE:
            raise  # let caller do fallback-to-search-URLs
        # Attempt 2: slightly slower read timeout
        data = _tavily_post(base, READ_TIMEOUT_SLOW)
        _cache_set(key, data); return data

def get_product_pricing_and_links(product_name: str, category: str) -> Dict[str, Any]:
    """One fast call (domain-filtered) + graceful fallback to vendor search URLs if Tavily is slow."""
    if not TAVILY_API_KEY:
        return {
            "price": None, "purchase_link": None, "available_stores": [],
            "error": "Tavily API key not configured",
        }
    try:
        q = f"{product_name} {category} price buy online"
        data = _tavily_search(q, include_domains=VENDOR_DOMAINS, max_results=MAX_RESULTS)

        # collect all retailer hits
        stores: Dict[str, Dict[str, Any]] = {}
        for r in data.get("results", []):
            url = r.get("url") or ""
            dom = _domain(url)
            if not any(dom.endswith(vd) for vd in VENDOR_DOMAINS):
                continue
            name = _vendor_from_domain(dom)
            price_str = _parse_price(r.get("content") or "")
            prev = stores.get(dom)
            if (not prev) or (_pdp_score(url) > _pdp_score(prev["url"])):
                stores[dom] = {"name": name, "url": url, "price": price_str}

        available = sorted(stores.values(), key=lambda s: (s["price"] is None, -_pdp_score(s["url"])))
        purchase_link = available[0]["url"] if available else None
        best_price = next((s["price"] for s in available if s["price"]), None)

        return {
            "price": best_price,
            "purchase_link": purchase_link,
            "available_stores": available,
            "search_results": len(data.get("results", [])),
            "error": None,
        }

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
        # Fallback: vendor search pages so UI still renders & user can click out
        q = quote_plus(product_name)
        fallback = [
            {"name": "amazon",  "url": f"https://www.amazon.com/s?k={q}"},
            {"name": "bestbuy", "url": f"https://www.bestbuy.com/site/searchpage.jsp?st={q}"},
            {"name": "walmart", "url": f"https://www.walmart.com/search?q={q}"},
            {"name": "target",  "url": f"https://www.target.com/s?searchTerm={q}"},
            {"name": "newegg",  "url": f"https://www.newegg.com/p/pl?d={q}"},
        ]
        return {
            "price": None,
            "purchase_link": None,
            "available_stores": fallback,
            "search_results": 0,
            "error": "tavily_timeout",
        }
    except requests.exceptions.RequestException as e:
        return {"price": None, "purchase_link": None, "available_stores": [], "error": f"API request failed: {e}"}
    except Exception as e:
        return {"price": None, "purchase_link": None, "available_stores": [], "error": f"Unexpected error: {e}"}




