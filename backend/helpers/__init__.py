# ========== Helpers for agent nodes ==========
import math
from statistics import median
from urllib.parse import urlparse

TRUSTED_VENDORS = {"amazon", "bestbuy", "walmart", "target", "newegg", "bhphoto", "microcenter"}

PRICE_RX = [
    r"\$[\d,]+(?:\.\d+)?",
    r"[\d,]+(?:\.\d+)?\s?(?:usd|dollars?)",
    r"€\s?[\d\.]+", r"£\s?[\d\.]+", r"(?:inr|₹)\s?[\d,]+(?:\.\d+)?",
]

def _domain_to_vendor(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
        parts = host.split(".")
        domain = ".".join(parts[-2:]) if len(parts) >= 2 else host
        if domain.endswith("amazon.com"): return "amazon"
        if domain.endswith("bestbuy.com"): return "bestbuy"
        if domain.endswith("walmart.com"): return "walmart"
        if domain.endswith("target.com"): return "target"
        if domain.endswith("newegg.com"): return "newegg"
        if domain.endswith("bhphotovideo.com"): return "bhphoto"
        if domain.endswith("microcenter.com"): return "microcenter"
        return domain or "unknown"
    except Exception:
        return "unknown"

def _parse_price_string(s: str | None) -> tuple[float | None, str]:
    if not s: return None, "USD"
    s_low = s.lower()
    currency = "USD"
    if "€" in s or "eur" in s_low: currency = "EUR"
    elif "£" in s or "gbp" in s_low: currency = "GBP"
    elif "inr" in s_low or "₹" in s_low: currency = "INR"
    m = None
    for rx in PRICE_RX:
        m = re.search(rx, s_low, flags=re.IGNORECASE)
        if m: break
    if not m: return None, currency
    digits = re.sub(r"[^\d.]", "", m.group())
    try:
        return float(digits), currency
    except Exception:
        return None, currency

def _build_offers_from_pricing(pricing: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    """Turn your `pricing` shape into comparable offers list."""
    if not pricing: return []
    offers: list[Dict[str, Any]] = []
    # primary purchase_link + price
    if pricing.get("purchase_link"):
        price_val, currency = _parse_price_string(pricing.get("price"))
        offers.append({
            "vendor": _domain_to_vendor(pricing["purchase_link"]),
            "url": pricing["purchase_link"],
            "price": price_val,
            "currency": currency,
        })
    # additional stores (may not have price)
    for s in pricing.get("available_stores", []):
        url = s.get("url"); 
        if not url: continue
        if any(o["url"] == url for o in offers): 
            continue
        # try to parse a price if Tavily attached one
        p = s.get("price") if isinstance(s, dict) else None
        price_val, currency = _parse_price_string(p) if p else (None, "USD")
        offers.append({
            "vendor": _domain_to_vendor(url),
            "url": url,
            "price": price_val,
            "currency": currency,
        })
    return offers

def _pick_best_offer(offers: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not offers: return None
    priced = [o for o in offers if isinstance(o.get("price"), (int, float))]
    if priced:
        return min(priced, key=lambda o: o["price"])
    # prefer recognizable vendors
    priority = ["amazon", "bestbuy", "walmart", "target", "newegg", "bhphoto", "microcenter"]
    offers.sort(key=lambda o: priority.index(o["vendor"]) if o["vendor"] in priority else 999)
    return offers[0]

def _category_anchor(category: str) -> float:
    """Very rough 'fair price' anchors per category; tune as you grow."""
    c = (category or "").lower()
    if "espresso" in c: return 600.0
    if "headphone" in c: return 300.0
    if "backpack" in c: return 180.0
    return 500.0

def _median_price(offers: list[Dict[str, Any]]) -> float | None:
    vals = [o["price"] for o in offers if isinstance(o.get("price"), (int, float))]
    return median(vals) if vals else None
