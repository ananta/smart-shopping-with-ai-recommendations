"""
Main entry point for the smart shopping recommendation demo.

This script loads a small set of mock Reddit data from a JSON file and
exposes a simple agentic workflow built with LangGraph.  A user's goal
is first analysed to extract high‑level shopping categories.  The
application then looks up the most upvoted product comments from the
mock community data for each category and finally synthesises a
friendly recommendation summary.  This demonstrates how you could
build an end‑to‑end RAG‑style application using structured community
content without scraping live data.

Dependencies:
  - langgraph
  - langchain-openai
  - langchain-core
  - openai
  Ensure that your OpenAI API key is available in the environment as
  OPENAI_API_KEY before running this script.

To run the example interactively, execute:

    python main.py

It will print a sample recommendation summary for an example user goal.
"""

import json
import os
from typing import TypedDict, List, Dict, Any, Tuple
import re
import math
import time

from langchain_openai import ChatOpenAI
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    # If langchain_openai is not installed, fallback to None.  Semantic
    # retrieval will gracefully return a helpful message.
    OpenAIEmbeddings = None
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from services.tavily import get_product_pricing_and_links, _safe_json, fetch_quick_specs_snippets, _vendor_from_url
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Data loading and indexing
# -----------------------------------------------------------------------------

# The mock data describes three typical community recommendation scenarios.
# Each entry contains a question, a list of comment objects and metadata.
# We read the file specified by the DATA_FILE environment variable (default
# ``mock_data.json``).  When you swap in real Reddit data later, set
# DATA_FILE in your environment to the appropriate filename.

load_dotenv()
DATA_FILE = os.getenv("DATA_FILE", "mock_data.json")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    reddit_data: List[Dict[str, Any]] = json.load(f)






# --- Pricing helpers ---
from decimal import Decimal
from urllib.parse import urlparse, urlencode, urlsplit, urlunsplit, parse_qsl


CURRENCY_MAP = {"$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP", "₹": "INR", "inr": "INR"}
VENDOR_BY_DOMAIN = {
    "amazon.com": "amazon", "bestbuy.com": "bestbuy", "walmart.com": "walmart",
    "target.com": "target", "newegg.com": "newegg"
}
PRICE_PATTERNS = [
    r"\$[\d,]+(?:\.\d+)?", r"[\d,]+(?:\.\d+)?\s?(?:usd|dollars?)",
    r"€\s?[\d\.]+", r"£\s?[\d\.]+", r"(?:inr|₹)\s?[\d,]+(?:\.\d+)?",
]

def _domain_to_vendor(url: str) -> str:
    try:
        host = urlparse(url).netloc.split(":")[0].lower()
        parts = host.split(".")
        domain = ".".join(parts[-2:]) if len(parts) >= 2 else host
        return VENDOR_BY_DOMAIN.get(domain, domain or "unknown")
    except Exception:
        return "unknown"

def _parse_price_string(s: str | None) -> tuple[float | None, str]:
    if not s:
        return None, "USD"
    s_low = s.lower()
    currency = "USD"
    for k, v in CURRENCY_MAP.items():
        if k in s_low:
            currency = v; break
    m = None
    for rx in PRICE_PATTERNS:
        m = re.search(rx, s_low, flags=re.IGNORECASE)
        if m: break
    if not m: return None, currency
    digits = re.sub(r"[^\d.]", "", m.group())
    try:
        return float(Decimal(digits)), currency
    except Exception:
        return None, currency

# optional: trivial affiliate tagger
AFFILIATE_TAG = os.getenv("AFFILIATE_TAG", "")  # e.g., "dartpick-20"
def to_affiliate(url: str) -> str:
    if not AFFILIATE_TAG or "amazon.com" not in url.lower():  # example impl
        return url
    scheme, netloc, path, query, frag = urlsplit(url)
    q = dict(parse_qsl(query))
    q["tag"] = AFFILIATE_TAG
    return urlunsplit((scheme, netloc, path, urlencode(q), frag))




# -----------------------------------------------------------------------------
# State definition
# -----------------------------------------------------------------------------

class ToolkitState(TypedDict):
    """Shared state used by the agentic graph.

    Attributes:
      user_goal: The raw free‑form goal provided by the user (e.g., "I want to
        build a home espresso setup under $500").
      category_plan: A list of high‑level shopping categories derived from the
        user goal (e.g., ["espresso machine", "grinder"]).
      recommendations: A mapping of category names to recommended products
        returned by the retrieval component.
      final_output: The final natural language recommendation summary.
    """

    user_goal: str
    category_plan: List[str]
    recommendations: Dict[str, Any]
    final_output: str
    spec: Dict[str, Any]




# ---------- AI: spec_builder ----------
spec_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You convert a shopping goal into a structured spec.\n"
     "Return STRICT JSON with keys: budget(number|null), priorities(string[]), "
     "constraints(object), persona(string).\n"
     "constraints may include booleans like milk_drinks, beginner, quiet, portability.\n"
     "If unknown, use null or []. Output ONLY JSON."),
    ("human", "Goal:\n{goal}")
])

def ai_spec_builder(state: ToolkitState) -> ToolkitState:
    goal = state.get("user_goal","")
    # call LLM
    chain = spec_prompt | llm
    resp = chain.invoke({"goal": goal})
    spec = _safe_json(resp.content, {"budget": None, "priorities": [], "constraints": {}, "persona": ""})
    # budget fallback (cheap, robust)
    m = re.search(r"\$?\s*([0-9]{2,5})\b", goal.replace(",", ""))
    if (spec.get("budget") is None) and m:
        try: spec["budget"] = float(m.group(1))
        except: pass
    # normalize booleans
    cons = spec.get("constraints") or {}
    for k in ["milk_drinks","beginner","quiet","portability"]:
        if k in cons: cons[k] = bool(cons[k])
    state["spec"] = {"budget": spec.get("budget"), **cons, "priorities": spec.get("priorities", []), "persona": spec.get("persona","")}
    return state



# ---------- AI: attribute_extractor ----------
attr_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract the most decision-relevant attributes for ANY product category "
     "(e.g., electronics, kitchen, luggage, apparel, photography, home, tools). "
     "Focus on facts users need to compare/buy.\n\n"
     "Guidance (non-exhaustive): performance, size/weight/dimensions, capacity, "
     "compatibility & standards (ports, mounts, voltage, OS/support), power/battery, "
     "materials/durability, warranty/support, included accessories.\n\n"
     "Return STRICT JSON ONLY:\n"
     "{{\n"
     "  \"attributes\": [\n"
     "    {{\"name\":\"string\",\"value\":(string|number|boolean|null),\"unit\":\"string|null\",\"confidence\":0.0-1.0}}\n"
     "  ],\n"
     "  \"citations\": [\"string\"]\n"
     "}}\n"
     "Rules: 6–10 items max. If unknown, use null. Do not invent. Prefer concise, normalized values "
     "(e.g., \"1.2 kg\", unit:\"kg\"; \"65 Wh\", unit:\"Wh\"; \"USB-C PD\", unit:null)."
    ),
    ("human",
     "Product: {product}\nCategory: {category}\n\n"
     "Reddit evidence:\n{reddit}\n\n"
     "Web snippets:\n{snippets}"
    ),
])

def _build_reddit_evidence(comments: List[Dict[str, Any]], limit:int=2) -> str:
    lines=[]
    for c in comments[:limit]:
        lines.append(f"- ({c.get('votes',0)} upvotes) {c.get('comment','')[:400]}")
    return "\n".join(lines) if lines else "None"

def _slug_attr(name: str) -> str:
    return re.sub(r"[^a-z0-9]+","_", name.lower()).strip("_")


def ai_attribute_extractor(state: ToolkitState) -> ToolkitState:
    recs = state.get("recommendations", {})
    enriched: Dict[str, Any] = {}

    def extract_for_item(product: str, category: str, reddit_block: str) -> Dict[str, Any]:
        snippets = fetch_quick_specs_snippets(product, category, max_results=2)
        chain = attr_prompt | llm
        resp = chain.invoke({
            "product": product,
            "category": category,
            "reddit": reddit_block,
            "snippets": json.dumps(snippets)[:1800],
        })
        parsed = _safe_json(resp.content, {"attributes": [], "citations": []})
        attrs: List[Dict[str, Any]] = parsed.get("attributes") or []
        # Build a convenience map for UI/synth: {"weight": {"value": "...","unit":"..."}}
        attrs_map: Dict[str, Any] = {}
        for a in attrs:
            n = _slug_attr(str(a.get("name","")).strip() or "attr")
            if n and n not in attrs_map:
                attrs_map[n] = {k: a.get(k) for k in ("value","unit","confidence")}
        return {"attributes": attrs, "attributes_map": attrs_map, "citations": parsed.get("citations", [])}

    for cat, info in recs.items():
        if isinstance(info, dict) and "top_comments" in info:
            out_list = []
            for c in info.get("top_comments", []):
                product = c.get("product") or ""
                reddit_txt = _build_reddit_evidence([c], limit=1)
                extracted = extract_for_item(product, cat, reddit_txt)
                out_list.append({**c, **extracted})
            enriched[cat] = {"top_comments": out_list}
        elif isinstance(info, dict) and info.get("product"):
            product = info.get("product") or ""
            reddit_txt = _build_reddit_evidence([info], limit=1)
            extracted = extract_for_item(product, cat, reddit_txt)
            enriched[cat] = {**info, **extracted}
        else:
            enriched[cat] = info

    return {**state, "recommendations": enriched}


# ---------- AI: decision_maker ----------
decision_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a shopping agent. Decide whether to 'buy', 'wait', or 'consider alt' for ANY product type.\n"
     "Use evidence: user spec (budget/priorities/constraints), attributes (extracted facts), price context, offers, and retailers.\n"
     "Heuristics: prefer trusted retailers when obvious from URLs; avoid too-good-to-be-true prices; respect user's budget/needs.\n"
     "Return STRICT JSON ONLY:\n"
     "{{\n"
     "  \"decision\": \"buy|wait|consider alt\",\n"
     "  \"why\": \"string (<=160 chars)\",\n"
     "  \"factors\": [\"string\"],\n"
     "  \"confidence\": 0.0\n"
     "}}\n"
     "Be concise and evidence-based. If signals conflict or are missing, lower confidence and explain briefly."
    ),
    ("human",
     "User spec:\n{spec}\n\n"
     "Product: {product}\nCategory: {category}\n\n"
     "Attributes (list):\n{attributes}\n\n"
     "Attributes (map):\n{attributes_map}\n\n"
     "Price context:\n{price}\n\n"
     "Offers (subset):\n{offers}\n\n"
     "Retailers:\n{retailers}\n"
    ),
])

def _num_from_price_str(s: str | None) -> float | None:
    if not s: return None
    # keep digits and dot, drop currency symbols/words
    m = re.search(r"[\d][\d,\.]*", s)
    if not m: return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None

def _best_price_from_item(item: Dict[str, Any]) -> tuple[float | None, str | None, Dict[str, Any] | None]:
    """
    Returns (best_price_number, currency, best_offer_dict) for a rec item.
    Priority:
      1) item.best_offer.{effective_price|price}
      2) min(offers[].price)
      3) parse pricing.price string
    """
    # 1) best_offer
    best = item.get("best_offer") or {}
    # accept int/float directly; parse strings via _parse_price_string
    price_val = None
    currency = None
    if isinstance(best, dict):
        raw = best.get("effective_price", best.get("price"))
        if isinstance(raw, (int, float)):
            price_val = float(raw); currency = best.get("currency") or "USD"
        elif isinstance(raw, str):
            price_val, currency = _parse_price_string(raw)
        else:
            currency = best.get("currency") or None

    # 2) scan offers if still missing
    if price_val is None:
        offers = item.get("offers") or []
        priced = [(o.get("price"), o.get("currency") or "USD", o)
                  for o in offers if isinstance(o.get("price"), (int, float))]
        if priced:
            price_val, currency, best = min(priced, key=lambda x: x[0])

    # 3) fallback to the raw pricing blob string
    if price_val is None:
        pricing_blob = item.get("pricing") or {}
        if isinstance(pricing_blob, dict) and pricing_blob.get("price"):
            price_val, currency = _parse_price_string(pricing_blob.get("price"))

    return price_val, (currency or "USD"), (best or None)


def _quick_guards(spec: Dict[str, Any], attrs_map: Dict[str, Any], best_price: float | None):
    guards = []
    if best_price is not None and spec.get("budget") is not None and best_price > float(spec["budget"]):
        guards.append(f"over_budget_by={best_price - float(spec['budget']):.2f}")
    volt = str((attrs_map.get("voltage") or {}).get("value") or "").lower()
    if volt and any(v in volt for v in ["220", "230"]) and "us" in (spec.get("persona","").lower()):
        guards.append("voltage_mismatch")
    return guards


def ai_decision_maker(state: ToolkitState) -> ToolkitState:
    spec = state.get("spec", {}) or {}
    recs = state.get("recommendations", {}) or {}
    decided: Dict[str, Any] = {}


    def decide_one(item: Dict[str, Any], category: str) -> Dict[str, Any]:
            product = item.get("product") or ""


            # ---- attributes: handle list or dict; build both forms for the prompt ----
            attrs_list = item.get("attributes") or []
            attrs_map = item.get("attributes_map") or {}
            if isinstance(attrs_list, dict):
                # legacy dict -> mirror to list
                attrs_map = attrs_list
                attrs_list = [{"name": k, "value": v} for k, v in attrs_list.items()]

            # ---- pricing context: accept best_offer/offers or raw pricing blob ----
            pricing_blob = item.get("pricing") or {}
            offers = item.get("offers") or []
            best = item.get("best_offer") or {}

            # Compute a numeric best price first
            best_price = None
            currency = None
            if isinstance(best, dict):
                p = best.get("effective_price", best.get("price"))
                if isinstance(p, (int, float)):
                    best_price = float(p)
                    currency = best.get("currency") or "USD"
                elif isinstance(p, str):
                    # parse string like "$399.99"
                    val = _num_from_price_str(p)
                    if val is not None:
                        best_price = val
                        currency = best.get("currency") or "USD"

            # If still missing, take the lowest numeric price from offers[]
            if best_price is None and isinstance(offers, list):
                priced = [(o.get("price"), o.get("currency") or "USD") 
                          for o in offers if isinstance(o.get("price"), (int, float))]
                if priced:
                    priced.sort(key=lambda t: t[0])
                    best_price, currency = priced[0]

            # Final fallback: parse raw pricing string
            if best_price is None and isinstance(pricing_blob, dict) and pricing_blob.get("price"):
                best_price = _num_from_price_str(pricing_blob.get("price"))
                currency = currency or "USD"

            # ✅ Now compute guard flags (needs best_price)
            guard_flags = _quick_guards(spec, attrs_map, best_price)

            # offers summary for the prompt (limit to a few)
            offers_summary: List[Dict[str, Any]] = []
            for o in (offers[:4] if isinstance(offers, list) else []):
                offers_summary.append({
                    "vendor": o.get("vendor"),
                    "price": o.get("effective_price", o.get("price")),
                    "coupon": o.get("coupon_applied"),
                    "risk": o.get("risk_score"),
                    "url": o.get("url"),
                })

            # retailer URLs from pricing blob as extra signal
            retailers: List[Dict[str, Any]] = []
            if isinstance(pricing_blob, dict):
                if pricing_blob.get("purchase_link"):
                    retailers.append({"url": pricing_blob["purchase_link"]})
                for s in pricing_blob.get("available_stores", [])[:5]:
                    if isinstance(s, dict) and s.get("url"):
                        retailers.append({"url": s["url"]})
                    elif isinstance(s, str):
                        retailers.append({"url": s})

            # compact price context (include guards so the LLM sees them)
            price_ctx = {"best_price": best_price, "currency": currency, "guards": guard_flags}

            chain = decision_prompt | llm
            resp = chain.invoke({
                "spec": json.dumps(spec),
                "product": product,
                "category": category,
                "attributes": json.dumps(attrs_list),
                "attributes_map": json.dumps(attrs_map),
                "price": json.dumps(price_ctx),
                "offers": json.dumps(offers_summary),
                "retailers": json.dumps(retailers),
            })
            out = _safe_json(resp.content, {
                "decision": "consider alt",
                "why": "Not enough information.",
                "factors": [],
                "confidence": 0.3
            })

            return {**item, "agent_decision": out, "guards": guard_flags, "best_price": best_price, "currency": currency}

    for cat, info in recs.items():
        if isinstance(info, dict) and "top_comments" in info:
            decided[cat] = {"top_comments": [decide_one(c, cat) for c in info.get("top_comments", [])]}
        elif isinstance(info, dict) and info.get("product"):
            decided[cat] = decide_one(info, cat)
        else:
            decided[cat] = info

    return {**state, "recommendations": decided}



# Multi offer list function

def offers_from_tavily(product_name: str, category: str) -> list[dict]:
    """
    Calls your existing get_product_pricing_and_links once,
    returns a list of comparable offers: [{vendor, url, price, currency}, ...]
    """
    data = get_product_pricing_and_links(product_name, category)

    print(data)

    offers: list[dict] = []
    # primary hit (purchase_link + price)
    if data.get("purchase_link"):
        price_val, currency = _parse_price_string(data.get("price"))
        offers.append({
            "vendor": _domain_to_vendor(data["purchase_link"]),
            "url": data["purchase_link"],
            "price": price_val,
            "currency": currency,
            "affiliate_url": to_affiliate(data["purchase_link"])
        })

    # secondary stores (may not have price yet)
    for store in data.get("available_stores", [])[:5]:
        url = store.get("url"); 
        if not url: continue
        # skip duplicates
        if any(o["url"] == url for o in offers): 
            continue
        offers.append({
            "vendor": _domain_to_vendor(url),
            "url": url,
            "price": None,
            "currency": "USD",
            "affiliate_url": to_affiliate(url)
        })

    # always return at least one placeholder to keep shape stable
    return offers or [{"vendor": "unknown", "url": "", "price": None, "currency": "USD"}]



# Pricing enricher
def pricing_enricher(state: ToolkitState) -> ToolkitState:
    recs = state.get("recommendations", {})
    enriched = {}

    for cat, info in recs.items():
        # Shape A: {"top_comments": [ {product, comment, votes}, ... ]}
        if isinstance(info, dict) and "top_comments" in info:
            new_list = []
            for c in info.get("top_comments", []):
                offers = offers_from_tavily(product_name=c["product"], category=cat)
                # choose best offer (lowest known price first, else first vendor)
                priced = [o for o in offers if o.get("price") is not None]
                best = min(priced, key=lambda o: o["price"]) if priced else (offers[0] if offers else None)
                new_list.append({**c, "offers": offers, "best_offer": best})
            enriched[cat] = {"top_comments": new_list}

        # Shape B (naïve retriever): {"product": ..., "comment": ..., "votes": ...}
        elif isinstance(info, dict) and info.get("product"):
            offers = offers_from_tavily(product_name=info["product"], category=cat)
            priced = [o for o in offers if o.get("price") is not None]
            best = min(priced, key=lambda o: o["price"]) if priced else (offers[0] if offers else None)
            enriched[cat] = {**info, "offers": offers, "best_offer": best}

        else:
            enriched[cat] = info  # passthrough (e.g., message only)

    return {**state, "recommendations": enriched}


# Build an in‑memory index: map each lowercased category to the list of
# comment entries.  Each comment entry carries a product name, the free
# text comment and an upvote count.  The mock data file uses the key
# ``category`` to store a single category name for each post; this
# implementation normalises the names to lowercase for easy lookup.
category_index: Dict[str, List[Dict[str, Any]]] = {}
for post in reddit_data:
    cat = post.get("category")
    if cat:
        key = cat.lower().strip()
        category_index.setdefault(key, []).extend(post.get("context", []))

# Derive a comma‑separated list of available categories for the prompt.
available_categories = ", ".join(cat.title() for cat in category_index.keys())


# -----------------------------------------------------------------------------
# LLM and prompt setup
# -----------------------------------------------------------------------------

# Instantiate a ChatOpenAI model.  It will automatically read the API key
# from the OPENAI_API_KEY environment variable.  Adjust the temperature as
# needed for more or less creative responses.
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Instantiate an embedding model for semantic retrieval, if available.  If
# ``OpenAIEmbeddings`` is not importable (e.g. ``langchain_openai`` is not
# installed), ``embedding_model`` will be set to None.  The semantic
# retriever checks for ``None`` and falls back to a helpful message.
if OpenAIEmbeddings is not None:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
else:
    embedding_model = None

# Prompt for the goal analysis step.  It instructs the LLM to return a
# simple list of shopping categories relevant to the user goal.  We avoid
# bullet characters so that we can easily split the response by newlines.
# Build the analysis prompt dynamically based on the categories present in the data.
# The assistant is told to choose from the available categories and to return exactly
# those names as they appear in the list.  This allows new domains to be added
# simply by updating the data file without modifying the code.  We also instruct
# the model to output one category per line, without bullet characters or extra
# commentary, so that the response can be split reliably.
analyze_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"You are an assistant that maps user goals to one or more of the following "
        f"categories: {available_categories}. "
        "Read the user's goal and return the category names that best match. "
        "Output one category per line and use exactly the category names provided above."
    ),
    ("human", "{user_goal}")
])


def spec_builder(state: ToolkitState) -> ToolkitState:
    goal = state.get("user_goal", "")
    spec: Dict[str, Any] = {}
    # budget like "$500" or "under 700"
    m = re.search(r"\$?\s*([0-9]{2,5})\s*(?:bucks|dollars)?", goal.lower())
    if m:
        try: spec["budget"] = float(m.group(1))
        except: pass
    # simple intents
    text = goal.lower()
    spec["milk_drinks"] = any(w in text for w in ["latte", "cappuccino", "milk", "microfoam"])
    spec["beginner"] = any(w in text for w in ["beginner", "first", "starter", "easy"])
    spec["quiet"] = "quiet" in text or "low noise" in text
    state["spec"] = spec
    return state


def analyze_goal(state: ToolkitState) -> ToolkitState:
    """Analyse the user goal and derive a list of categories.

    Uses the LLM to identify which broad product categories are relevant to
    the stated goal.  The LLM is instructed to return each category on its
    own line without bullet points.  The list is stored on the state as
    `category_plan`.
    """
    chain = analyze_prompt | llm
    response = chain.invoke({"user_goal": state["user_goal"]})
    # Split into lines and remove empties
    categories: List[str] = [
        line.strip() for line in response.content.splitlines() if line.strip()
    ]
    return {
        **state,
        "category_plan": categories,
    }


# -----------------------------------------------------------------------------
# Retrieval logic
# -----------------------------------------------------------------------------

def _lookup_comments(category: str) -> List[Dict[str, Any]]:
    """Helper to normalise category names and return a list of comments.

    This function attempts to find comments for the exact category name.  If
    there is no exact match, it tries singular and plural variants (e.g.,
    "headphones" → "headphone" and vice versa).  It returns a list of
    comment dictionaries or an empty list if none are found.
    """
    key = category.lower().strip()
    comments = category_index.get(key, [])
    if not comments:
        # Try singular form (remove trailing 's')
        if key.endswith('s'):
            singular = key[:-1]
            comments = category_index.get(singular, [])
            if comments:
                return comments
        # Try plural form (add trailing 's')
        plural = key + 's'
        comments = category_index.get(plural, [])
    return comments

# A simple tokenizer used for lexical matching.  It splits on
# non‑alphanumeric characters and lowercases all tokens.  This helper is
# shared by lexical and MMR retrieval functions.
def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.split(r"\W+", text) if w]


def simple_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Naïve retriever that selects the single highest‑voted comment per category."""
    updated_recs = state.get("recommendations", {}).copy()
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            best_comment = max(comments, key=lambda c: c.get("votes", 0))
            
            # Get pricing and purchase links for the product
            pricing_info = get_product_pricing_and_links(best_comment["product"], category)
            
            updated_recs[category] = {
                "product": best_comment["product"],
                "comment": best_comment["comment"],
                "votes": best_comment["votes"],
                "pricing": pricing_info,
            }
        else:
            updated_recs[category] = {
                "product": None,
                "comment": None,
                "votes": 0,
                "pricing": None,
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }


def lexical_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Advanced retriever that ranks comments by lexical overlap with the user goal.

    For each category, this function tokenises the user's goal and each comment,
    counts the overlap of query words with the comment text and sorts comments
    by this overlap and then by vote count.  It returns up to two of the
    highest‑scoring comments per category.  This approach is useful when the
    query contains specific keywords (e.g., "noise cancelling") and we want to
    favour comments mentioning those terms.
    """
    updated_recs = state.get("recommendations", {}).copy()
    # Tokenise the user goal once for efficiency
    query_words = set(_tokenize(state.get("user_goal", "")))
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Define a scoring function: lexical overlap count
            def lex_score(comment: Dict[str, Any]) -> int:
                comment_words = set(_tokenize(comment.get("comment", "")))
                return len(query_words & comment_words)
            # Sort by lexical score then by votes
            sorted_comments = sorted(
                comments,
                key=lambda c: (lex_score(c), c.get("votes", 0)),
                reverse=True,
            )
            top_comments = sorted_comments[:2]
            recs = [
                {
                    "product": c.get("product"),
                    "comment": c.get("comment"),
                    "votes": c.get("votes", 0),
                }
                for c in top_comments
            ]
            updated_recs[category] = {
                "top_comments": recs,
            }
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }


def mmr_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Advanced retriever that applies a simple Maximal Marginal Relevance (MMR) selection.

    This strategy picks the comment with the highest lexical overlap with the query as
    the first recommendation.  For the second recommendation, it selects a comment
    that is both relevant (lexical overlap) and diverse relative to the first comment.
    Diversity is approximated by penalising comments whose word overlap with the
    first comment is large.  A lambda parameter controls the trade‑off between
    relevance and diversity (0.0 ≤ lambda ≤ 1.0).  The function returns up to two
    recommendations per category.
    """
    updated_recs = state.get("recommendations", {}).copy()
    query_words = set(_tokenize(state.get("user_goal", "")))
    lam = 0.7  # trade‑off parameter between relevance and diversity
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Define relevance score as lexical overlap count
            def rel_score(comment: Dict[str, Any]) -> int:
                return len(query_words & set(_tokenize(comment.get("comment", ""))))
            # Sort comments by relevance then by votes
            sorted_comments = sorted(
                comments,
                key=lambda c: (rel_score(c), c.get("votes", 0)),
                reverse=True,
            )
            # Pick the most relevant comment as the first selection
            first = sorted_comments[0]
            first_words = set(_tokenize(first.get("comment", "")))
            # Prepare to find a second comment based on MMR
            second = None
            best_mmr = float("-inf")
            for c in sorted_comments[1:]:
                c_words = set(_tokenize(c.get("comment", "")))
                relevance = rel_score(c)
                diversity_penalty = len(c_words & first_words)
                mmr_score = lam * relevance - (1 - lam) * diversity_penalty
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    second = c
            # Collect results
            recs = [
                {
                    "product": first.get("product"),
                    "comment": first.get("comment"),
                    "votes": first.get("votes", 0),
                }
            ]
            if second:
                recs.append(
                    {
                        "product": second.get("product"),
                        "comment": second.get("comment"),
                        "votes": second.get("votes", 0),
                    }
                )
            updated_recs[category] = {
                "top_comments": recs,
            }
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }

def advanced_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Advanced retriever that selects the top‑2 comments per category and aggregates them.

    This function returns a list of up to two comments for each category, sorted
    by vote count.  It provides more context for the synthesiser to work with
    and improves recall relative to the naïve retriever.
    """
    updated_recs = state.get("recommendations", {}).copy()
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Sort comments by vote count descending and take the top two
            top_comments = sorted(comments, key=lambda c: c.get("votes", 0), reverse=True)[:2]
            # Convert each comment to a structured dict
            recs = []
            for comment in top_comments:
                # Get pricing and purchase links for each product
                pricing_info = get_product_pricing_and_links(comment.get("product"), category)
                
                recs.append({
                    "product": comment.get("product"),
                    "comment": comment.get("comment"),
                    "votes": comment.get("votes", 0),
                    "pricing": pricing_info,
                })
            updated_recs[category] = {
                "top_comments": recs,
            }
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }


def tfidf_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Advanced retriever using TF‑IDF and cosine similarity.

    This retriever represents the user goal and each comment using a TF‑IDF
    vectorizer.  It then computes cosine similarity between the query vector
    and each comment vector and selects up to two comments with the highest
    similarity scores.  Vote counts are used as a secondary sort key.  This
    approach captures relevance beyond simple lexical overlap and does not
    require external embedding APIs.  If no comments exist for a category, a
    fallback message is returned.
    """
    updated_recs = state.get("recommendations", {}).copy()
    query = state.get("user_goal", "")
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Extract the comment texts
            texts = [c.get("comment", "") for c in comments]
            # Fit a TF‑IDF vectorizer on the comment corpus plus the query
            # We include the query to ensure that its vocabulary is represented.
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform(texts + [query])
            except Exception:
                # If the vectorizer fails (e.g. due to an empty vocabulary), fall back
                updated_recs[category] = {
                    "top_comments": [],
                    "message": "TF‑IDF vectorisation failed; consider simplifying your goal."
                }
                continue
            # The query vector is the last row
            query_vec = tfidf_matrix[-1]
            comment_vecs = tfidf_matrix[:-1]
            # Compute cosine similarity between query and each comment
            sims = cosine_similarity(query_vec, comment_vecs).flatten()
            # Prepare list of (similarity, votes, comment)
            scored = []
            for sim, c in zip(sims, comments):
                scored.append((sim, c.get("votes", 0), c))
            # Sort by similarity then votes
            scored.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
            top = scored[:2]
            recs = [
                {
                    "product": c.get("product"),
                    "comment": c.get("comment"),
                    "votes": c.get("votes", 0),
                }
                for (_, _, c) in top
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }


def semantic_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Advanced retriever that uses semantic similarity between the query and comments.

    For each category, this function embeds the user goal and each comment's text
    using an OpenAI embedding model.  It computes a simple dot‑product similarity
    between the query embedding and each comment embedding and selects up to two
    comments with the highest similarity scores (ties broken by vote counts).  This
    approach captures semantic relationships beyond exact keyword matches and can
    surface relevant recommendations even when the wording differs between the
    query and the comments.  If no comments are available for a category, a
    fallback message is returned.
    """
    updated_recs = state.get("recommendations", {}).copy()
    # If the embedding model is not available (e.g. langchain_openai not installed),
    # return a fallback message for each category.
    if embedding_model is None:
        for category in state.get("category_plan", []):
            updated_recs[category] = {
                "top_comments": [],
                "message": "Semantic retriever unavailable: langchain_openai is not installed."
            }
        return {**state, "recommendations": updated_recs}
    # Compute the embedding for the entire user goal once per invocation
    user_goal = state.get("user_goal", "")
    try:
        query_emb = embedding_model.embed_query(user_goal)
    except Exception:
        # If embedding fails (e.g. due to API key issues), return fallback messages.
        for category in state.get("category_plan", []):
            updated_recs[category] = {
                "top_comments": [],
                "message": "Semantic retriever unavailable: failed to embed query."
            }
        return {**state, "recommendations": updated_recs}
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            scored: List[Any] = []
            for comment in comments:
                comment_text = comment.get("comment", "")
                try:
                    comment_emb = embedding_model.embed_query(comment_text)
                except Exception:
                    # Skip comments that fail to embed
                    continue
                # Compute dot‑product similarity between embeddings
                similarity = 0.0
                for a, b in zip(query_emb, comment_emb):
                    similarity += a * b
                scored.append((similarity, comment))
            if scored:
                # Sort by similarity score then by votes
                scored.sort(key=lambda tup: (tup[0], tup[1].get("votes", 0)), reverse=True)
                top_two = scored[:2]
                recs = [
                    {
                        "product": c.get("product"),
                        "comment": c.get("comment"),
                        "votes": c.get("votes", 0),
                    }
                    for (_, c) in top_two
                ]
                updated_recs[category] = {"top_comments": recs}
            else:
                updated_recs[category] = {
                    "top_comments": [],
                    "message": "No comments could be embedded for semantic retrieval."
                }
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }

# -----------------------------------------------------------------------------
# Additional advanced retrieval strategies
#
# The following retrievers implement a variety of more sophisticated
# approaches that can be valuable in different information retrieval
# scenarios.  These are not strictly necessary for the mock dataset, but
# demonstrate how one could incorporate techniques such as BM25 ranking,
# query rewriting, parent document retrieval and ensemble scoring.  Each
# function follows the same input/output pattern used by the earlier
# retrieval methods: it accepts a ToolkitState and returns an updated
# ToolkitState with a `recommendations` mapping that holds either a list
# of top comment objects (`"top_comments"`) or a single product entry.

# Precompute IDF values for BM25 across all comments.  We build a global
# vocabulary and document frequency table keyed by token.  This table is
# used by the BM25 retriever to compute scores.  Each category has its
# own set of comments, so the IDF must be computed per category.
def _build_bm25_stats(comments: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute inverse document frequency (IDF) for each token in a list of comments.

    Args:
        comments: A list of comment dictionaries for a single category.

    Returns:
        A mapping from token to IDF value.
    """
    # Compute document frequencies per token
    df: Dict[str, int] = {}
    N = len(comments)
    for c in comments:
        words = set(_tokenize(c.get("comment", "")))
        for w in words:
            df[w] = df.get(w, 0) + 1
    # Compute IDF with a smoothing factor (0.5) to avoid division by zero
    idf: Dict[str, float] = {}
    for w, count in df.items():
        idf[w] = math.log( (N - count + 0.5) / (count + 0.5) + 1.0 )
    return idf

def bm25_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Retrieve comments using a simplified BM25 ranking function.

    For each category, this retriever computes a BM25 score for every
    comment based on term frequency and inverse document frequency of
    tokens relative to the user goal.  It returns up to two comments
    per category ranked by the BM25 score and then by vote count.  A
    smoothing factor is used to avoid zero divisions.  This method is
    useful for scenarios where term weighting and document length
    normalisation improve retrieval quality compared to simple lexical
    overlap.
    """
    updated_recs = state.get("recommendations", {}).copy()
    query_tokens = _tokenize(state.get("user_goal", ""))
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Build IDF table for this category
            idf = _build_bm25_stats(comments)
            # Average document length for normalisation
            avgdl = 0.0
            lengths = []
            for c in comments:
                tokens = _tokenize(c.get("comment", ""))
                lengths.append(len(tokens))
            if lengths:
                avgdl = sum(lengths) / len(lengths)
            else:
                avgdl = 1.0
            k1 = 1.5
            b = 0.75
            # Score each comment using BM25
            scored = []
            for c in comments:
                tokens = _tokenize(c.get("comment", ""))
                tf_counts: Dict[str, int] = {}
                for t in tokens:
                    tf_counts[t] = tf_counts.get(t, 0) + 1
                doc_len = len(tokens) if tokens else 1
                score = 0.0
                for t in query_tokens:
                    tf = tf_counts.get(t, 0)
                    # Only score terms present in the comment
                    if tf > 0:
                        idf_t = idf.get(t, 0.0)
                        # BM25 term score
                        score += idf_t * ( (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl)) )
                scored.append((score, c.get("votes", 0), c))
            # Sort by BM25 score then votes
            scored.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
            top = scored[:2]
            recs = [
                {
                    "product": c.get("product"),
                    "comment": c.get("comment"),
                    "votes": c.get("votes", 0),
                }
                for (_, _, c) in top
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }

def compression_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Retrieve comments using a simple compression scheme.

    This strategy compresses each comment down to a shorter representation
    consisting of the product name and the first 50 characters of the
    comment.  It then applies lexical overlap scoring (similar to the
    lexical retriever) on these compressed strings and returns up to
    two top comments per category.  Although simplistic, this approach
    illustrates how context compression could be integrated into a
    retrieval pipeline.  In a real system you might use an LLM or
    summarisation algorithm to create more meaningful compressed
    representations.
    """
    updated_recs = state.get("recommendations", {}).copy()
    query_tokens = set(_tokenize(state.get("user_goal", "")))
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            compressed = []
            for c in comments:
                text = c.get("comment", "")
                comp = f"{c.get('product', '')}: {text[:50]}"
                compressed.append((comp, c))
            # Compute lexical overlap on compressed text
            def comp_score(item: Tuple[str, Dict[str, Any]]) -> int:
                comp_text, _ = item
                words = set(_tokenize(comp_text))
                return len(query_tokens & words)
            # Sort by score then votes
            sorted_items = sorted(
                compressed,
                key=lambda tup: (comp_score(tup), tup[1].get("votes", 0)),
                reverse=True,
            )
            top_items = sorted_items[:2]
            recs = [
                {
                    "product": item[1].get("product"),
                    "comment": item[1].get("comment"),
                    "votes": item[1].get("votes", 0),
                }
                for item in top_items
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }

# Synonym dictionary for simple query expansion used in multi‑query retrieval.
_synonym_map: Dict[str, List[str]] = {
    "best": ["top", "great", "excellent"],
    "budget": ["cheap", "affordable", "low cost"],
    "noise": ["anc", "noise cancel", "noise-cancelling"],
    "cancelling": ["canceling", "anc"],
    "headphones": ["headphone", "cans", "earphones"],
    "lightweight": ["ultralight", "ultra light", "light"],
    "backpack": ["pack", "rucksack", "bag"],
    "multi": ["multi-day", "long"],
    "espresso": ["coffee", "espresso machine", "barista"],
}

def _expand_query(query: str) -> List[str]:
    """Generate multiple query variants based on simple synonyms.

    Tokenises the input query and substitutes words with their synonyms
    according to a predefined map.  Generates a set of expanded query
    strings including the original.  This is a rudimentary form of
    multi‑query retrieval that can help surface comments using different
    terminology.

    Args:
        query: The original user query.

    Returns:
        A list of query strings.
    """
    tokens = _tokenize(query)
    variants: List[List[str]] = [[]]
    for t in tokens:
        synonyms = _synonym_map.get(t, [])
        new_variants = []
        for v in variants:
            # Include the original token
            new_variants.append(v + [t])
            # Include each synonym
            for syn in synonyms:
                new_variants.append(v + [syn])
        variants = new_variants
    # Reconstruct strings and deduplicate
    queries: List[str] = []
    for variant in variants:
        q = " ".join(variant)
        if q not in queries:
            queries.append(q)
    return queries

def multiquery_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Retrieve comments using a multi‑query expansion technique.

    For each category, this retriever generates multiple query variants
    based on simple synonym substitution.  It then computes a lexical
    overlap score for each comment across all query variants and
    aggregates the scores (e.g., by summing).  Comments with higher
    aggregated scores are considered more relevant.  The top two
    comments per category are returned.  This technique can increase
    recall when users phrase goals differently from the community.
    """
    updated_recs = state.get("recommendations", {}).copy()
    # Generate expanded queries
    queries = _expand_query(state.get("user_goal", ""))
    # Precompute tokenised variants
    query_tokens_list = [set(_tokenize(q)) for q in queries]
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            scored = []
            for c in comments:
                c_words = set(_tokenize(c.get("comment", "")))
                # Aggregate lexical overlap across query variants
                total_overlap = 0
                for q_tokens in query_tokens_list:
                    total_overlap += len(c_words & q_tokens)
                scored.append((total_overlap, c.get("votes", 0), c))
            # Sort by aggregated overlap then votes
            scored.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
            top_two = scored[:2]
            recs = [
                {
                    "product": c.get("product"),
                    "comment": c.get("comment"),
                    "votes": c.get("votes", 0),
                }
                for (_, _, c) in top_two
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }

def parent_document_retriever(state: ToolkitState) -> ToolkitState:
    """Retrieve recommendations by ranking entire categories as parent documents.

    Instead of scoring individual comments first, this method concatenates
    all comments in a category into a single parent document.  It then
    computes lexical overlap between the user goal and the parent
    documents to select up to two relevant categories.  For each
    selected category, the top‑voted comment is returned.  This
    approach is useful when categories are broad and we want to filter
    them before inspecting individual comments.
    """
    updated_recs = state.get("recommendations", {}).copy()
    query_tokens = set(_tokenize(state.get("user_goal", "")))
    # Build parent documents for each category
    parent_docs: List[Tuple[int, str, List[Dict[str, Any]]]] = []
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            concatenated = " ".join(c.get("comment", "") for c in comments)
            parent_docs.append((len(query_tokens & set(_tokenize(concatenated))), category, comments))
        else:
            parent_docs.append((0, category, []))
    # Sort categories by lexical overlap
    parent_docs.sort(key=lambda tup: tup[0], reverse=True)
    # Select up to two categories
    selected = parent_docs[:2]
    for overlap, category, comments in selected:
        if comments:
            # Pick the top voted comment within the category
            best_comment = max(comments, key=lambda c: c.get("votes", 0))
            updated_recs[category] = {
                "top_comments": [
                    {
                        "product": best_comment.get("product"),
                        "comment": best_comment.get("comment"),
                        "votes": best_comment.get("votes", 0),
                    }
                ]
            }
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }

def ensemble_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Combine multiple retrieval scores to select comments.

    This ensemble method computes both lexical overlap and TF‑IDF cosine
    similarity for each comment.  Scores from each metric are
    normalised to the range [0,1] and then averaged.  Comments with
    higher ensemble scores are preferred.  Ties are broken using vote
    counts.  This strategy aims to balance exact keyword matches and
    distributional similarity to improve retrieval robustness.
    """
    updated_recs = state.get("recommendations", {}).copy()
    query = state.get("user_goal", "")
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Compute lexical overlap scores
            query_tokens = set(_tokenize(query))
            lex_scores: List[float] = []
            for c in comments:
                c_words = set(_tokenize(c.get("comment", "")))
                lex_scores.append(float(len(query_tokens & c_words)))
            max_lex = max(lex_scores) if lex_scores else 1.0
            # Compute TF‑IDF scores
            texts = [c.get("comment", "") for c in comments]
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform(texts + [query])
            except Exception:
                # Fall back to lexical scores only
                combined = [
                    (lex_scores[i] / max_lex if max_lex else 0.0, c.get("votes", 0), c)
                    for i, c in enumerate(comments)
                ]
                combined.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
                top = combined[:2]
                recs = [
                    {
                        "product": c.get("product"),
                        "comment": c.get("comment"),
                        "votes": c.get("votes", 0),
                    }
                    for (_, _, c) in top
                ]
                updated_recs[category] = {"top_comments": recs}
                continue
            query_vec = tfidf_matrix[-1]
            comment_vecs = tfidf_matrix[:-1]
            sims = cosine_similarity(query_vec, comment_vecs).flatten()
            max_sim = max(sims) if sims.size > 0 else 1.0
            # Combine normalised lexical and TF‑IDF scores
            combined_scores: List[Tuple[float, int, Dict[str, Any]]] = []
            for i, c in enumerate(comments):
                lex_norm = (lex_scores[i] / max_lex) if max_lex else 0.0
                sim_norm = (sims[i] / max_sim) if max_sim else 0.0
                ensemble_score = 0.5 * lex_norm + 0.5 * sim_norm
                combined_scores.append((ensemble_score, c.get("votes", 0), c))
            combined_scores.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
            top_two = combined_scores[:2]
            recs = [
                {
                    "product": c.get("product"),
                    "comment": c.get("comment"),
                    "votes": c.get("votes", 0),
                }
                for (_, _, c) in top_two
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list."
            }
    return {
        **state,
        "recommendations": updated_recs,
    }


# -----------------------------------------------------------------------------
# Synthesis logic
# -----------------------------------------------------------------------------

# Prompt for the final recommendation summary.  It instructs the LLM to
# produce a concise, helpful explanation of why each recommended product
# matches the user's goal.  The list of recommendations is presented as
# plain text; the LLM uses that to generate a user‑facing summary.
synth_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant that crafts a friendly recommendation summary for a shopping toolkit."
    ),
    (
        "human",
        "User Goal: {user_goal}\n\n"
        "Recommendations:\n{recommendations}\n\n"
        "Explain briefly why each recommended product is suitable for the user's goal."
    ),
])


def synthesizer(state: ToolkitState) -> ToolkitState:
    lines = []
    for cat, info in state.get("recommendations", {}).items():
        if isinstance(info, dict) and "top_comments" in info:
            items = []
            for c in info.get("top_comments", []):
                best = c.get("best_offer")
                alt_cnt = max(0, len(c.get("offers", [])) - 1)
                price_str = f" @ ${best['price']:.2f} from {best['vendor']}" if (best and best.get("price")) else ""
                alt_str = f" (+{alt_cnt} more stores)" if alt_cnt else ""

                dec = c.get("agent_decision") or {}
                decision_str = ""
                if isinstance(dec, dict) and dec.get("decision"):
                    why = dec.get("why")
                    decision_str = f" | Decision: {dec.get('decision')}{(' – ' + why) if why else ''}"

                items.append(
                    f"{c['product']} (votes: {c['votes']}) – {c['comment']}{price_str}{alt_str}{decision_str}"
                )
            lines.append(f"{cat}: " + "; ".join(items) if items else f"{cat}: {info.get('message','No recommendations found.')}")
        elif isinstance(info, dict) and info.get("product"):
            best = info.get("best_offer")
            alt_cnt = max(0, len(info.get("offers", [])) - 1)
            price_str = f" @ ${best['price']:.2f} from {best['vendor']}" if (best and best.get("price")) else ""
            alt_str = f" (+{alt_cnt} more stores)" if alt_cnt else ""
            lines.append(f"{cat}: {info['product']} (votes: {info['votes']}) – {info['comment']}{price_str}{alt_str}")
        else:
            lines.append(f"{cat}: {info.get('message', 'No recommendations found.') if isinstance(info, dict) else str(info)}")

    recommendations_str = "\n".join(lines)
    chain = synth_prompt | llm
    response = chain.invoke({"user_goal": state["user_goal"], "recommendations": recommendations_str})
    return {**state, "final_output": response.content}





# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------

# Construct the LangGraph workflow.  We define each node using our
# previously declared functions and connect them in sequence: analysis →
# retrieval → synthesis.  The output of each node is passed as the state
# input to the next.
builder = StateGraph(ToolkitState)
builder.add_node("goal_analyzer", analyze_goal)

# Choose the retrieval strategy based on an environment variable.  If
# RETRIEVER_STRATEGY is set to "advanced", the graph will use the
# advanced_reddit_retriever; otherwise, it defaults to the simple version.
retriever_strategy = os.getenv("RETRIEVER_STRATEGY", "naive").lower()
# Determine which retriever node to add based on the strategy.  We check
# each option explicitly in order of precedence.  The default (fallback)
# is the naïve single comment retriever.
if retriever_strategy in ("semantic", "embedding"):
    # Use semantic similarity based on OpenAI embeddings
    builder.add_node("retriever", semantic_reddit_retriever)
elif retriever_strategy == "advanced":
    # Use the top‑2 vote‑based retriever
    builder.add_node("retriever", advanced_reddit_retriever)
elif retriever_strategy == "lexical":
    # Use lexical overlap to rank comments
    builder.add_node("retriever", lexical_reddit_retriever)
elif retriever_strategy == "mmr":
    # Use MMR (maximal marginal relevance) to select diverse recommendations
    builder.add_node("retriever", mmr_reddit_retriever)
elif retriever_strategy in ("tfidf", "vectorizer"):
    # Use a TF‑IDF vectoriser and cosine similarity for retrieval
    builder.add_node("retriever", tfidf_reddit_retriever)
elif retriever_strategy in ("bm25",):
    # Use BM25 scoring on tokenised comments
    builder.add_node("retriever", bm25_reddit_retriever)
elif retriever_strategy in ("compression", "compress"):
    # Use a compressed representation of comments for retrieval
    builder.add_node("retriever", compression_reddit_retriever)
elif retriever_strategy in ("multiquery", "multi", "multi-query"):
    # Use multi‑query expansion with simple synonyms
    builder.add_node("retriever", multiquery_reddit_retriever)
elif retriever_strategy in ("parent", "parentdocument", "parent_doc", "parent-document"):
    # Use parent document retrieval (aggregated category documents)
    builder.add_node("retriever", parent_document_retriever)
elif retriever_strategy == "ensemble":
    # Use an ensemble of lexical and TF‑IDF scores
    builder.add_node("retriever", ensemble_reddit_retriever)
else:
    # Default to naïve top‑1 vote‑based retriever
    builder.add_node("retriever", simple_reddit_retriever)

builder.add_node("synthesizer", synthesizer)

builder.add_node("pricing_enricher", pricing_enricher)


builder.add_node("ai_spec_builder", ai_spec_builder)
builder.add_node("ai_attribute_extractor", ai_attribute_extractor)
builder.add_node("ai_decision_maker", ai_decision_maker)

# Define the order of execution: analysis → retrieval → synthesis
builder.set_entry_point("goal_analyzer")
builder.add_edge("goal_analyzer", "ai_spec_builder")
builder.add_edge("ai_spec_builder", "retriever")
builder.add_edge("retriever", "pricing_enricher")
builder.add_edge("pricing_enricher", "ai_attribute_extractor")
builder.add_edge("ai_attribute_extractor", "ai_decision_maker")
builder.add_edge("ai_decision_maker", "synthesizer")
builder.set_finish_point("synthesizer")

graph = builder.compile()

# -----------------------------------------------------------------------------
# FastAPI integration
#
# Expose the LangGraph workflow via an HTTP API.  When the `/recommend`
# endpoint is called with a JSON payload containing a user goal, the
# application will invoke the graph and return the derived categories,
# recommended products and a concise summary.  This endpoint can be run
# locally using Uvicorn (e.g. `uvicorn main:app --reload`).
# -----------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Instantiate the FastAPI application
app = FastAPI()

# Add CORS middleware to allow requests from React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GoalRequest(BaseModel):
    """Request schema for the recommendation endpoint."""
    user_goal: str

@app.post("/recommend")
def recommend(req: GoalRequest):
    """Recommend products based on a user goal.

    This endpoint accepts a JSON body with a single field, `user_goal`,
    describing what the user wants (e.g. "I want to set up a budget home
    espresso station").  The function then runs the agentic graph to
    determine the relevant categories and community recommendations and
    returns them along with a natural language summary.
    """
    initial_state: ToolkitState = {
        "user_goal": req.user_goal,
        "category_plan": [],
        "recommendations": {},
        "final_output": "",
        "spec": {}
    }
    result_state = graph.invoke(initial_state)
    return {
        "user_goal": req.user_goal,
        "categories": result_state.get("category_plan", []),
        "recommendations": result_state.get("recommendations", {}),
        "summary": result_state.get("final_output", ""),
    }

def _public_slice(state: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the big state to just what the UI needs at each step (now includes pricing/offers)."""
    recs = state.get("recommendations", {})

    def _offer_lite(o: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "vendor": o.get("vendor"),
            "price": o.get("price"),          # numeric if known, else None
            "currency": o.get("currency") or "USD",
            "url": o.get("url"),
            "affiliate_url": o.get("affiliate_url"),
            "coupon": o.get("coupon"),        # may be None
            "risk": o.get("risk"),            # may be None (alias for risk_score)
        }

    def _best_lite(o: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not isinstance(o, dict):
            return None
        return _offer_lite(o)

    def _item_lite(c: Dict[str, Any]) -> Dict[str, Any]:
        # Keep minimal but useful comparison fields
        return {
            "product": c.get("product"),
            "votes": c.get("votes"),
            "best_price": c.get("best_price"),          # computed in ai_decision_maker
            "currency": c.get("currency") or "USD",
            "agent_decision": (c.get("agent_decision") or {}).get("decision"),
            "best_offer": _best_lite(c.get("best_offer")),
            "offers": [_offer_lite(o) for o in (c.get("offers") or [])],
        }

    def _lite(rec):
        if isinstance(rec, dict) and "top_comments" in rec:
            return {"top_comments": [_item_lite(c) for c in rec.get("top_comments", [])]}
        elif isinstance(rec, dict) and rec.get("product"):
            # single-item shape (naïve retriever)
            return _item_lite(rec)
        return rec

    return {
        "category_plan": state.get("category_plan", []),
        "spec": state.get("spec", {}),
        "recommendations": {k: _lite(v) for k, v in recs.items()},
        "final_output": state.get("final_output", None),
    }


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/recommend/stream")
def recommend_stream(user_goal: str):
    """
    Server-Sent Events stream of the graph execution.
    One event per node completion + a final 'done'.
    """
    initial_state: ToolkitState = {
        "user_goal": user_goal,
        "category_plan": [],
        "recommendations": {},
        "final_output": "",
        "spec": {}
    }

    def gen():
        # Let the client know we started
        yield _sse("start", {"user_goal": user_goal, "ts": time.time()})

        # Stream LangGraph node updates
        for update in graph.stream(initial_state):
            # update is a dict like {"node_name": <partial/new state>}
            for node_name, node_state in update.items():
                yield _sse(
                    "node_end",
                    {
                        "node": node_name,
                        "state": _public_slice(node_state),
                        "ts": time.time(),
                    },
                )

        # Finished
        # You can send the final state again if you want a definitive payload
        final_state = graph.invoke(initial_state)  # optional: if you want the full final
        yield _sse("done", {"state": _public_slice(final_state), "ts": time.time()})

    return StreamingResponse(gen(), media_type="text/event-stream")

def run_demo(user_goal: str) -> str:
    """Invoke the graph for a given user goal and return the final summary."""
    initial_state: ToolkitState = {
        "user_goal": user_goal,
        "category_plan": [],
        "recommendations": {},
        "final_output": "",
        "spec": {}
    }
    result_state = graph.invoke(initial_state)
    return result_state["final_output"]


if __name__ == "__main__":
    # A simple manual test.  You can change the input sentence to any
    # plausible goal, for example:
    #   "I want to set up a budget home espresso station"
    #   "Looking for good noise cancelling headphones for travel"
    #   "Need an ultralight backpack for thru‑hiking"
    demo_goal = "I want to set up a budget home espresso station"
    print("Input:", demo_goal)
    print("\nRecommendation Summary:\n")
    print(run_demo(demo_goal))
