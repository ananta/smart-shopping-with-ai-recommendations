import re
from decimal import Decimal
from urllib.parse import urlparse
from typing import List, Dict, Optional

CURRENCY_MAP = {"$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP", "₹": "INR"}

PRICE_REGEXES = [
    r"\$[\d,]+(?:\.\d+)?",                      # $1,234.56
    r"[\d,]+(?:\.\d+)?\s?(?:usd|dollars?)",     # 1,234.56 USD
    r"€\s?[\d\.]+",                             # €999.99
    r"£\s?[\d\.]+",                             # £399.99
    r"(?:inr|₹)\s?[\d,]+(?:\.\d+)?",            # ₹ 12,999
]

VENDOR_BY_DOMAIN = {
    "amazon.com": "amazon",
    "bestbuy.com": "bestbuy",
    "walmart.com": "walmart",
    "target.com": "target",
    "newegg.com": "newegg",
}


def parse_price_string(s: Optional[str]) -> (Optional[Decimal], str):
    """Return (amount, currency) from a free-form price string."""
    if not s:
        return None, "USD"
    s = s.strip().lower()
    # detect currency symbol/code
    currency = "USD"
    for k, v in CURRENCY_MAP.items():
        if k.lower() in s:
            currency = v
            break
    # extract numeric
    m = None
    for rx in PRICE_REGEXES:
        m = re.search(rx, s, flags=re.IGNORECASE)
        if m:
            break
    if not m:
        return None, currency
    digits = re.sub(r"[^\d\.]", "", m.group())  # keep numbers and dot
    if digits.count(".") > 1:
        # fall back: remove all dots except last
        parts = digits.split(".")
        digits = "".join(parts[:-1]).replace(".", "") + "." + parts[-1]
    try:
        return Decimal(digits), currency
    except Exception:
        return None, currency


def domain_to_vendor(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        # strip subdomain
        parts = host.split(":")[0].split(".")
        domain = ".".join(parts[-2:]) if len(parts) >= 2 else host
        return VENDOR_BY_DOMAIN.get(domain, domain or "unknown")
    except Exception:
        return "unknown"

