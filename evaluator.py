"""
Evaluation script for the smart shopping recommendation prototype.

This script compares the naïve and advanced retrieval strategies used in
the smart shopping prototype on a small synthetic test set.  It uses simple heuristics to
approximate RAGAS‑style metrics (faithfulness, response relevance, context
precision and context recall) without requiring external evaluation
libraries.  The goal is to provide a quantitative baseline to aid in
debugging and improvement.

Usage:

    python evaluator.py

The script loads the same ``mock_data.json`` file used by the application, runs
both retrieval strategies (naïve top‑1 and advanced top‑2) for each test
query and computes approximate metrics.  Results are printed to stdout.
"""

import sys
import re
from dotenv import load_dotenv
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    # If langchain_openai is not installed, set to None.  The semantic retriever
    # will gracefully fall back when embedding_model is None.
    OpenAIEmbeddings = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

# NOTE: We reimplement the retrieval logic here instead of importing
# ``main`` because the main module depends on packages (langchain_openai,
# fastapi, etc.) that may not be available when running this script.  To
# evaluate retrieval in isolation, we read the mock data file directly and
# implement the same helper functions that main.py uses.

import json
import os

# -----------------------------------------------------------------------------
# Data loading and indexing
# -----------------------------------------------------------------------------

load_dotenv()
# Determine which data file to use.  If the caller has set ``DATA_FILE`` in
# the environment then use that, otherwise default to ``mock_data.json``.
DATA_FILE = os.getenv("DATA_FILE", "mock_data.json")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    reddit_data: List[Dict[str, Any]] = json.load(f)

# Build an in‑memory index from category to comments.  The index maps the
# lower‑cased category name to a list of comment objects, each containing
# ``product``, ``comment`` and ``votes``.
category_index: Dict[str, List[Dict[str, Any]]] = {}
for post in reddit_data:
    cat = post.get("category")
    if cat:
        key = cat.lower().strip()
        category_index.setdefault(key, []).extend(post.get("context", []))

def _lookup_comments(category: str) -> List[Dict[str, Any]]:
    """Normalise category names and return a list of comments.

    Attempts to find comments for the exact category name; if none exist it
    tries singular and plural variants.  Returns an empty list if no
    matching category is found.
    """
    key = category.lower().strip()
    comments = category_index.get(key, [])
    if not comments:
        # Try singular form (remove trailing 's')
        if key.endswith("s"):
            singular = key[:-1]
            comments = category_index.get(singular, [])
            if comments:
                return comments
        # Try plural form (add trailing 's')
        plural = key + "s"
        comments = category_index.get(plural, [])
    return comments

# -----------------------------------------------------------------------------
# Embedding model setup for semantic retrieval
# -----------------------------------------------------------------------------

# Instantiate an embedding model once for the entire evaluation.  If
# ``OpenAIEmbeddings`` is unavailable (e.g. because ``langchain_openai`` is not
# installed), ``embedding_model`` will be set to None and the semantic
# retriever will produce fallback messages.
if OpenAIEmbeddings is not None:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
else:
    embedding_model = None

def simple_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Naïve retriever that selects the single highest‑voted comment per category."""
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            best_comment = max(comments, key=lambda c: c.get("votes", 0))
            updated_recs[category] = {
                "product": best_comment.get("product"),
                "comment": best_comment.get("comment", ""),
                "votes": best_comment.get("votes", 0),
            }
        else:
            updated_recs[category] = {
                "product": None,
                "comment": None,
                "votes": 0,
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {
        **state,
        "recommendations": updated_recs,
    }


# -----------------------------------------------------------------------------
# Additional advanced retrievers for evaluation
# -----------------------------------------------------------------------------

def lexical_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retriever that ranks comments by lexical overlap with the user's goal.

    For each category this function counts the overlap between the query
    words and the words in each comment.  It returns up to two comments
    with the highest overlap counts (ties broken by vote count).  The
    returned structure matches that of the advanced retriever (a
    ``top_comments`` list).  This retriever is useful when query keywords
    should be matched explicitly in the comments.
    """
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    # Tokenise user goal once
    query_words = set(tokenize(state.get("user_goal", "")))
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Lexical overlap score function
            def lex_score(comment: Dict[str, Any]) -> int:
                words = set(tokenize(comment.get("comment", "")))
                return len(query_words & words)
            sorted_comments = sorted(
                comments,
                key=lambda c: (lex_score(c), c.get("votes", 0)),
                reverse=True,
            )
            top_comments = sorted_comments[:2]
            recs = [
                {
                    "product": c.get("product"),
                    "comment": c.get("comment", ""),
                    "votes": c.get("votes", 0),
                }
                for c in top_comments
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {**state, "recommendations": updated_recs}


def mmr_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retriever that selects two comments using a simple Maximal Marginal Relevance (MMR) strategy.

    It first chooses the comment with the highest lexical overlap with the query.
    For the second comment it balances relevance (overlap with the query) against
    diversity (less overlap with the first comment).  The lambda parameter
    controls the trade‑off between relevance and diversity.
    """
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    query_words = set(tokenize(state.get("user_goal", "")))
    lam = 0.7
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Relevance score: lexical overlap count
            def rel_score(comment: Dict[str, Any]) -> int:
                return len(query_words & set(tokenize(comment.get("comment", ""))))
            # Sort comments by relevance then votes
            sorted_comments = sorted(
                comments,
                key=lambda c: (rel_score(c), c.get("votes", 0)),
                reverse=True,
            )
            # First comment: most relevant
            first = sorted_comments[0]
            first_words = set(tokenize(first.get("comment", "")))
            # MMR selection for second comment
            second = None
            best_mmr_score = float("-inf")
            for c in sorted_comments[1:]:
                c_words = set(tokenize(c.get("comment", "")))
                relevance = rel_score(c)
                diversity_penalty = len(c_words & first_words)
                mmr_score = lam * relevance - (1 - lam) * diversity_penalty
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    second = c
            recs = [
                {
                    "product": first.get("product"),
                    "comment": first.get("comment", ""),
                    "votes": first.get("votes", 0),
                }
            ]
            if second:
                recs.append(
                    {
                        "product": second.get("product"),
                        "comment": second.get("comment", ""),
                        "votes": second.get("votes", 0),
                    }
                )
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {**state, "recommendations": updated_recs}


def semantic_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retriever that ranks comments by semantic similarity using embeddings.

    For each category this function embeds the user's goal and each comment's
    text via an OpenAI embedding model and computes a dot‑product similarity
    score.  It returns up to two comments with the highest similarity scores
    (ties broken by vote counts).  If the embedding API call fails (e.g., due
    to missing API key), this function returns an empty recommendation list
    with an explanatory message.  The structure of the returned data matches
    that of other advanced retrievers: a ``top_comments`` list per category.
    """
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    user_goal = state.get("user_goal", "")
    # If embedding_model is None, semantics retrieval is unavailable
    if embedding_model is None:
        for category in state.get("category_plan", []):
            updated_recs[category] = {
                "top_comments": [],
                "message": "Semantic retriever unavailable: langchain_openai is not installed."
            }
        return {**state, "recommendations": updated_recs}
    try:
        query_emb = embedding_model.embed_query(user_goal)
    except Exception:
        # If embedding fails (e.g. due to API error), return fallback messages for each category
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
            for c in comments:
                comment_text = c.get("comment", "")
                try:
                    comment_emb = embedding_model.embed_query(comment_text)
                except Exception:
                    continue
                # Compute dot product (cosine similarity without normalisation)
                similarity = 0.0
                for a, b in zip(query_emb, comment_emb):
                    similarity += a * b
                scored.append((similarity, c))
            if scored:
                scored.sort(key=lambda tup: (tup[0], tup[1].get("votes", 0)), reverse=True)
                top_comments = scored[:2]
                recs = [
                    {
                        "product": c.get("product"),
                        "comment": c.get("comment", ""),
                        "votes": c.get("votes", 0),
                    }
                    for (_, c) in top_comments
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
    return {**state, "recommendations": updated_recs}


def tfidf_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retriever that ranks comments by TF‑IDF cosine similarity.

    It represents the user's goal and each comment's text using a TF‑IDF
    vectoriser from scikit‑learn, computes cosine similarity between the
    query vector and each comment vector, and returns up to two comments
    with the highest similarity scores (ties broken by vote count).  This
    method does not require external API calls and serves as a more
    semantic retrieval than simple lexical overlap.
    """
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    query = state.get("user_goal", "")
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            texts = [c.get("comment", "") for c in comments]
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform(texts + [query])
            except Exception:
                updated_recs[category] = {
                    "top_comments": [],
                    "message": "TF‑IDF vectorisation failed; consider simplifying your goal."
                }
                continue
            query_vec = tfidf_matrix[-1]
            comment_vecs = tfidf_matrix[:-1]
            sims = cosine_similarity(query_vec, comment_vecs).flatten()
            scored = []
            for sim, c in zip(sims, comments):
                scored.append((sim, c.get("votes", 0), c))
            scored.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
            top = scored[:2]
            recs = [
                {
                    "product": c.get("product"),
                    "comment": c.get("comment", ""),
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
    return {**state, "recommendations": updated_recs}
def advanced_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced retriever that selects the top‑2 comments per category and aggregates them."""
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            top_comments = sorted(comments, key=lambda c: c.get("votes", 0), reverse=True)[:2]
            recs = []
            for c in top_comments:
                recs.append({
                    "product": c.get("product"),
                    "comment": c.get("comment", ""),
                    "votes": c.get("votes", 0),
                })
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {
        **state,
        "recommendations": updated_recs,
    }


# -----------------------------------------------------------------------------
# Semantic retriever for evaluation
# -----------------------------------------------------------------------------

# Try to import the embedding model.  If this fails (e.g. because
# ``langchain_openai`` is not installed or no OpenAI API key is available),
# semantic retrieval will be skipped.  In that case, the semantic retriever
# returns fallback messages.
try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
    _embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception:
    _embedding_model = None


def semantic_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retriever that uses semantic similarity between the query and comments.

    This function embeds the user's goal and each comment using an OpenAI
    embedding model.  It computes a simple dot‑product similarity between the
    query and each comment and selects up to two comments with the highest
    similarity scores.  If the embedding model is unavailable, it returns a
    fallback message for each category.
    """
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    # If no embedding model is available, return fallback messages
    if _embedding_model is None:
        for category in state.get("category_plan", []):
            updated_recs[category] = {
                "top_comments": [],
                "message": "Semantic retriever unavailable in this environment.",
            }
        return {**state, "recommendations": updated_recs}
    # Compute the query embedding once
    user_goal = state.get("user_goal", "")
    try:
        query_emb = _embedding_model.embed_query(user_goal)
    except Exception:
        for category in state.get("category_plan", []):
            updated_recs[category] = {
                "top_comments": [],
                "message": "Semantic retrieval failed due to embedding error.",
            }
        return {**state, "recommendations": updated_recs}
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            scored: List[Any] = []
            for c in comments:
                comment_text = c.get("comment", "")
                try:
                    comment_emb = _embedding_model.embed_query(comment_text)
                except Exception:
                    continue
                # Compute dot product similarity
                sim = 0.0
                for a, b in zip(query_emb, comment_emb):
                    sim += a * b
                scored.append((sim, c))
            if scored:
                scored.sort(key=lambda tup: (tup[0], tup[1].get("votes", 0)), reverse=True)
                top_two = scored[:2]
                recs = [
                    {
                        "product": c.get("product"),
                        "comment": c.get("comment", ""),
                        "votes": c.get("votes", 0),
                    }
                    for (_, c) in top_two
                ]
                updated_recs[category] = {"top_comments": recs}
            else:
                updated_recs[category] = {
                    "top_comments": [],
                    "message": "No comments could be embedded for semantic retrieval.",
                }
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {**state, "recommendations": updated_recs}


def tokenize(text: str) -> List[str]:
    """Simple tokenizer that splits on non‑alphabetic characters and lowercases."""
    return [w.lower() for w in re.split(r"\W+", text) if w]


def compute_metrics(
    retrieved: Any,
    query: str,
    category: str,
    expected_product: str,
    comments_dataset: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute RAGAS‑style metrics for a single query and retrieval result.

    Args:
        retrieved: Either a dict with keys ``product``, ``comment``, ``votes``
                  (for the naïve retriever) or a dict with a ``top_comments``
                  list (for the advanced retriever).
        query: The original user question.
        category: The category associated with the query.
        expected_product: The product we expect the model to recommend.
        comments_dataset: List of all comments for this category (from
                          ``category_index``).

    Returns:
        Dictionary with keys ``faithfulness``, ``relevance``, ``precision``
        and ``recall``.
    """

    # Convert query to a set of meaningful words
    query_words = set(tokenize(query))

    # Normalise expected product for comparison
    expected_product_norm = expected_product.strip().lower()

    # Determine which comments were retrieved and put them into a list for
    # uniform processing.  Each entry is a dict with ``product``, ``comment``.
    retrieved_comments: List[Dict[str, Any]] = []
    if retrieved is None:
        retrieved_comments = []
    elif isinstance(retrieved, dict) and "top_comments" in retrieved:
        # advanced case
        for c in retrieved.get("top_comments", []):
            retrieved_comments.append(
                {"product": c.get("product"), "comment": c.get("comment", "")}
            )
    elif isinstance(retrieved, dict) and retrieved.get("product"):
        retrieved_comments.append(
            {"product": retrieved.get("product"), "comment": retrieved.get("comment", "")}
        )

    # Gather relevant comments in the dataset (any comment recommending the expected product)
    all_relevant_comments = [
        c for c in comments_dataset if str(c.get("product", "")).lower() == expected_product_norm
    ]

    # Compute metrics
    faithfulness = 1.0 if retrieved_comments else 0.0
    # Faithfulness is set to 1 if we have at least one retrieved comment (i.e. no hallucination)

    # Relevance: 1.0 if any retrieved comment's product matches the expected product
    relevance = 0.0
    for rc in retrieved_comments:
        if str(rc.get("product", "")).lower() == expected_product_norm:
            relevance = 1.0
            break

    # Precision: average overlap ratio between query words and comment words
    precision_scores: List[float] = []
    for rc in retrieved_comments:
        comment_words = set(tokenize(rc.get("comment", "")))
        if comment_words:
            overlap = query_words & comment_words
            precision_scores.append(len(overlap) / len(comment_words))
    precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    # Recall: ratio of retrieved relevant comments to all relevant comments
    total_relevant = len(all_relevant_comments)
    if total_relevant > 0:
        retrieved_relevant = sum(
            1
            for rc in retrieved_comments
            if str(rc.get("product", "")).lower() == expected_product_norm
        )
        recall = retrieved_relevant / total_relevant
    else:
        recall = 0.0

    return {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "precision": precision,
        "recall": recall,
    }

# -----------------------------------------------------------------------------
# Additional retrieval strategies for evaluation
#
# To mirror the functionality in main.py, we implement BM25, compression,
# multi‑query, parent document and ensemble retrievals here.  These
# functions reuse helpers such as ``tokenize`` and ``_lookup_comments`` to
# produce comparable outputs for metric computation.

def _build_bm25_stats_eval(comments: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute IDF values for BM25 within a category for evaluation.

    Args:
        comments: List of comment objects in the same category.

    Returns:
        Dict mapping tokens to their IDF values.
    """
    df: Dict[str, int] = {}
    N = len(comments)
    for c in comments:
        words = set(tokenize(c.get("comment", "")))
        for w in words:
            df[w] = df.get(w, 0) + 1
    idf: Dict[str, float] = {}
    for w, count in df.items():
        # Add a smoothing factor 0.5 to avoid division by zero
        idf[w] = max(0.0, ( (N - count + 0.5) / (count + 0.5) ))
    return idf

def bm25_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve comments using a simplified BM25 ranking function (evaluation).

    Args:
        state: Dict with keys ``user_goal`` and ``category_plan``.

    Returns:
        Updated state with ``top_comments`` lists for each category.
    """
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    query_tokens = tokenize(state.get("user_goal", ""))
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            idf = _build_bm25_stats_eval(comments)
            # Compute average document length
            lengths = [len(tokenize(c.get("comment", ""))) for c in comments]
            avgdl = sum(lengths) / len(lengths) if lengths else 1.0
            k1 = 1.5
            b = 0.75
            scored = []
            for c in comments:
                words = tokenize(c.get("comment", ""))
                tf_counts: Dict[str, int] = {}
                for w in words:
                    tf_counts[w] = tf_counts.get(w, 0) + 1
                doc_len = len(words) if words else 1
                score = 0.0
                for t in query_tokens:
                    tf = tf_counts.get(t, 0)
                    if tf > 0:
                        idf_t = idf.get(t, 0.0)
                        score += idf_t * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl)))
                scored.append((score, c.get("votes", 0), c))
            scored.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
            top = scored[:2]
            recs = [
                {
                    "product": c.get("product"),
                    "comment": c.get("comment", ""),
                    "votes": c.get("votes", 0),
                }
                for (_, _, c) in top
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {**state, "recommendations": updated_recs}

def compression_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve comments using a simple compression technique (evaluation).

    This method shortens each comment to the product name and the first 50
    characters.  It then applies lexical overlap on the compressed text to
    rank comments and returns up to two comments per category.
    """
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    query_tokens = set(tokenize(state.get("user_goal", "")))
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Build compressed representations
            compressed: List[Tuple[str, Dict[str, Any]]] = []
            for c in comments:
                comp = f"{c.get('product', '')}: {c.get('comment', '')[:50]}"
                compressed.append((comp, c))
            # Compute lexical overlap score
            def comp_score(item: Tuple[str, Dict[str, Any]]) -> int:
                words = set(tokenize(item[0]))
                return len(query_tokens & words)
            sorted_items = sorted(
                compressed,
                key=lambda tup: (comp_score(tup), tup[1].get("votes", 0)),
                reverse=True,
            )
            top_items = sorted_items[:2]
            recs = [
                {
                    "product": item[1].get("product"),
                    "comment": item[1].get("comment", ""),
                    "votes": item[1].get("votes", 0),
                }
                for item in top_items
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {**state, "recommendations": updated_recs}

# Simple synonym map for query expansion (matches main.py)
_synonym_map_eval: Dict[str, List[str]] = {
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

def _expand_query_eval(query: str) -> List[str]:
    tokens = tokenize(query)
    variants: List[List[str]] = [[]]
    for t in tokens:
        syns = _synonym_map_eval.get(t, [])
        new_variants: List[List[str]] = []
        for v in variants:
            # Original token
            new_variants.append(v + [t])
            # Synonyms
            for s in syns:
                new_variants.append(v + [s])
        variants = new_variants
    # Deduplicate
    seen: set = set()
    queries = []
    for v in variants:
        q = " ".join(v)
        if q not in seen:
            seen.add(q)
            queries.append(q)
    return queries

def multiquery_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve comments using multi‑query expansion (evaluation)."""
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    queries = _expand_query_eval(state.get("user_goal", ""))
    query_tokens_list = [set(tokenize(q)) for q in queries]
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            scored = []
            for c in comments:
                c_words = set(tokenize(c.get("comment", "")))
                total_overlap = sum(len(c_words & q_tokens) for q_tokens in query_tokens_list)
                scored.append((total_overlap, c.get("votes", 0), c))
            scored.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
            top_two = scored[:2]
            recs = [
                {
                    "product": c.get("product"),
                    "comment": c.get("comment", ""),
                    "votes": c.get("votes", 0),
                }
                for (_, _, c) in top_two
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {**state, "recommendations": updated_recs}

def parent_document_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parent document retrieval strategy for evaluation.

    Concatenate all comments per category into a single text, compute lexical
    overlap with the user goal and select up to two categories.  For each
    selected category, return the top voted comment.
    """
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    query_tokens = set(tokenize(state.get("user_goal", "")))
    parent_docs: List[Tuple[int, str, List[Dict[str, Any]]]] = []
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            concatenated = " ".join(c.get("comment", "") for c in comments)
            overlap_count = len(query_tokens & set(tokenize(concatenated)))
            parent_docs.append((overlap_count, category, comments))
        else:
            parent_docs.append((0, category, []))
    parent_docs.sort(key=lambda tup: tup[0], reverse=True)
    selected = parent_docs[:2]
    for overlap, category, comments in selected:
        if comments:
            best_comment = max(comments, key=lambda c: c.get("votes", 0))
            updated_recs[category] = {
                "top_comments": [
                    {
                        "product": best_comment.get("product"),
                        "comment": best_comment.get("comment", ""),
                        "votes": best_comment.get("votes", 0),
                    }
                ]
            }
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {**state, "recommendations": updated_recs}

def ensemble_reddit_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """Ensemble retrieval using lexical and TF‑IDF scores (evaluation)."""
    updated_recs: Dict[str, Any] = state.get("recommendations", {}).copy()
    query = state.get("user_goal", "")
    for category in state.get("category_plan", []):
        comments = _lookup_comments(category)
        if comments:
            # Lexical scores
            query_tokens = set(tokenize(query))
            lex_scores: List[float] = []
            for c in comments:
                c_words = set(tokenize(c.get("comment", "")))
                lex_scores.append(float(len(query_tokens & c_words)))
            max_lex = max(lex_scores) if lex_scores else 1.0
            # TF‑IDF scores
            texts = [c.get("comment", "") for c in comments]
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform(texts + [query])
            except Exception:
                # Fall back to lexical only
                combined_scores = [
                    (lex_scores[i] / max_lex if max_lex else 0.0, c.get("votes", 0), c)
                    for i, c in enumerate(comments)
                ]
                combined_scores.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
                top = combined_scores[:2]
                recs = [
                    {
                        "product": c.get("product"),
                        "comment": c.get("comment", ""),
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
                    "comment": c.get("comment", ""),
                    "votes": c.get("votes", 0),
                }
                for (_, _, c) in top_two
            ]
            updated_recs[category] = {"top_comments": recs}
        else:
            updated_recs[category] = {
                "top_comments": [],
                "message": "No strong Reddit consensus found. Try rephrasing your goal or expanding the category list.",
            }
    return {**state, "recommendations": updated_recs}


def main_eval() -> None:
    """Run evaluation on a predefined test set and print results."""
    # Synthetic test set matching our mock data
    tests = [
        {
            "query": "What's the best espresso machine under $500?",
            "category": "Espresso Machines",
            "expected_product": "Gaggia Classic Pro",
        },
        {
            "query": "Best budget noise cancelling headphones?",
            "category": "Headphones",
            "expected_product": "Anker Soundcore Q30",
        },
        {
            "query": "What's a good lightweight hiking backpack for multi-day treks?",
            "category": "Hiking Gear",
            "expected_product": "Osprey Exos 48",
        },
    ]

    # Use the locally built category_index for computing recall
    global category_index

    # Evaluation accumulators for each strategy.  We accumulate the sum of each
    # metric across all test queries and then divide by the number of tests
    # at the end to produce averages.  New strategies such as BM25,
    # compression, multi‑query, parent document and ensemble are included
    # alongside the existing ones.
    metrics_naive_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_adv_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_lex_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_mmr_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_semantic_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_tfidf_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_bm25_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_comp_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_multi_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_parent_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}
    metrics_ens_total = {"faithfulness": 0, "relevance": 0, "precision": 0, "recall": 0}

    print("\nEvaluation Results:\n")
    header = [
        "Query",
        "Naïve (product)",
        "Naïve metrics (F/R/P/Rc)",
        "Advanced (products)",
        "Advanced metrics (F/R/P/Rc)",
        "Lexical (products)",
        "Lexical metrics (F/R/P/Rc)",
        "MMR (products)",
        "MMR metrics (F/R/P/Rc)",
        "Semantic (products)",
        "Semantic metrics (F/R/P/Rc)",
        "TF‑IDF (products)",
        "TF‑IDF metrics (F/R/P/Rc)",
        "BM25 (products)",
        "BM25 metrics (F/R/P/Rc)",
        "Compression (products)",
        "Compression metrics (F/R/P/Rc)",
        "Multi‑Query (products)",
        "Multi‑Query metrics (F/R/P/Rc)",
        "Parent (products)",
        "Parent metrics (F/R/P/Rc)",
        "Ensemble (products)",
        "Ensemble metrics (F/R/P/Rc)",
    ]
    print(", ".join(header))

    for t in tests:
        query = t["query"]
        category = t["category"]
        expected_product = t["expected_product"]
        comments_dataset = category_index.get(category.lower(), [])

        # Prepare a state for the retrievers.  We do not use the LLM, so
        # category_plan contains the correct category manually.
        state = {
            "user_goal": query,
            "category_plan": [category],
            "recommendations": {},
            "final_output": "",
        }

        # Naïve retrieval
        naive_state = simple_reddit_retriever(state)
        naive_rec = naive_state["recommendations"].get(category, {})
        naive_product = naive_rec.get("product") if isinstance(naive_rec, dict) else None
        naive_metrics = compute_metrics(
            naive_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_naive_total:
            metrics_naive_total[k] += naive_metrics[k]
        # Advanced retrieval
        adv_state = advanced_reddit_retriever(state)
        adv_rec = adv_state["recommendations"].get(category, {})
        adv_products = []
        if isinstance(adv_rec, dict) and "top_comments" in adv_rec:
            for c in adv_rec.get("top_comments", []):
                adv_products.append(c.get("product"))
        adv_metrics = compute_metrics(
            adv_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_adv_total:
            metrics_adv_total[k] += adv_metrics[k]

        # Lexical retrieval
        lex_state = lexical_reddit_retriever(state)
        lex_rec = lex_state["recommendations"].get(category, {})
        lex_products = []
        if isinstance(lex_rec, dict) and "top_comments" in lex_rec:
            for c in lex_rec.get("top_comments", []):
                lex_products.append(c.get("product"))
        lex_metrics = compute_metrics(
            lex_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_lex_total:
            metrics_lex_total[k] += lex_metrics[k]

        # MMR retrieval
        mmr_state = mmr_reddit_retriever(state)
        mmr_rec = mmr_state["recommendations"].get(category, {})
        mmr_products = []
        if isinstance(mmr_rec, dict) and "top_comments" in mmr_rec:
            for c in mmr_rec.get("top_comments", []):
                mmr_products.append(c.get("product"))
        mmr_metrics = compute_metrics(
            mmr_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_mmr_total:
            metrics_mmr_total[k] += mmr_metrics[k]

        # Semantic retrieval
        semantic_state = semantic_reddit_retriever(state)
        semantic_rec = semantic_state["recommendations"].get(category, {})
        semantic_products = []
        if isinstance(semantic_rec, dict) and "top_comments" in semantic_rec:
            for c in semantic_rec.get("top_comments", []):
                semantic_products.append(c.get("product"))
        semantic_metrics = compute_metrics(
            semantic_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_semantic_total:
            metrics_semantic_total[k] += semantic_metrics[k]

        # TF‑IDF retrieval
        tfidf_state = tfidf_reddit_retriever(state)
        tfidf_rec = tfidf_state["recommendations"].get(category, {})
        tfidf_products = []
        if isinstance(tfidf_rec, dict) and "top_comments" in tfidf_rec:
            for c in tfidf_rec.get("top_comments", []):
                tfidf_products.append(c.get("product"))
        tfidf_metrics = compute_metrics(
            tfidf_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_tfidf_total:
            metrics_tfidf_total[k] += tfidf_metrics[k]

        # BM25 retrieval
        bm25_state = bm25_reddit_retriever(state)
        bm25_rec = bm25_state["recommendations"].get(category, {})
        bm25_products = []
        if isinstance(bm25_rec, dict) and "top_comments" in bm25_rec:
            for c in bm25_rec.get("top_comments", []):
                bm25_products.append(c.get("product"))
        bm25_metrics = compute_metrics(
            bm25_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_bm25_total:
            metrics_bm25_total[k] += bm25_metrics[k]

        # Compression retrieval
        comp_state = compression_reddit_retriever(state)
        comp_rec = comp_state["recommendations"].get(category, {})
        comp_products = []
        if isinstance(comp_rec, dict) and "top_comments" in comp_rec:
            for c in comp_rec.get("top_comments", []):
                comp_products.append(c.get("product"))
        comp_metrics = compute_metrics(
            comp_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_comp_total:
            metrics_comp_total[k] += comp_metrics[k]

        # Multi‑Query retrieval
        multi_state = multiquery_reddit_retriever(state)
        multi_rec = multi_state["recommendations"].get(category, {})
        multi_products = []
        if isinstance(multi_rec, dict) and "top_comments" in multi_rec:
            for c in multi_rec.get("top_comments", []):
                multi_products.append(c.get("product"))
        multi_metrics = compute_metrics(
            multi_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_multi_total:
            metrics_multi_total[k] += multi_metrics[k]

        # Parent document retrieval
        parent_state = parent_document_retriever(state)
        parent_rec = parent_state["recommendations"].get(category, {})
        parent_products = []
        if isinstance(parent_rec, dict) and "top_comments" in parent_rec:
            for c in parent_rec.get("top_comments", []):
                parent_products.append(c.get("product"))
        parent_metrics = compute_metrics(
            parent_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_parent_total:
            metrics_parent_total[k] += parent_metrics[k]

        # Ensemble retrieval
        ens_state = ensemble_reddit_retriever(state)
        ens_rec = ens_state["recommendations"].get(category, {})
        ens_products = []
        if isinstance(ens_rec, dict) and "top_comments" in ens_rec:
            for c in ens_rec.get("top_comments", []):
                ens_products.append(c.get("product"))
        ens_metrics = compute_metrics(
            ens_rec, query, category, expected_product, comments_dataset
        )
        for k in metrics_ens_total:
            metrics_ens_total[k] += ens_metrics[k]

        # Format metrics for printing
        def fmt(m: Dict[str, float]) -> str:
            return f"{m['faithfulness']:.2f}/{m['relevance']:.2f}/{m['precision']:.2f}/{m['recall']:.2f}"

        naive_metrics_str = fmt(naive_metrics)
        adv_metrics_str = fmt(adv_metrics)
        lex_metrics_str = fmt(lex_metrics)
        mmr_metrics_str = fmt(mmr_metrics)
        semantic_metrics_str = fmt(semantic_metrics)
        tfidf_metrics_str = fmt(tfidf_metrics)
        bm25_metrics_str = fmt(bm25_metrics)
        comp_metrics_str = fmt(comp_metrics)
        multi_metrics_str = fmt(multi_metrics)
        parent_metrics_str = fmt(parent_metrics)
        ens_metrics_str = fmt(ens_metrics)

        print(
            f"{query}, {naive_product}, {naive_metrics_str}, "
            f"{'; '.join([p for p in adv_products if p])}, {adv_metrics_str}, "
            f"{'; '.join([p for p in lex_products if p])}, {lex_metrics_str}, "
            f"{'; '.join([p for p in mmr_products if p])}, {mmr_metrics_str}, "
            f"{'; '.join([p for p in semantic_products if p])}, {semantic_metrics_str}, "
            f"{'; '.join([p for p in tfidf_products if p])}, {tfidf_metrics_str}, "
            f"{'; '.join([p for p in bm25_products if p])}, {bm25_metrics_str}, "
            f"{'; '.join([p for p in comp_products if p])}, {comp_metrics_str}, "
            f"{'; '.join([p for p in multi_products if p])}, {multi_metrics_str}, "
            f"{'; '.join([p for p in parent_products if p])}, {parent_metrics_str}, "
            f"{'; '.join([p for p in ens_products if p])}, {ens_metrics_str}"
        )

    # Compute averages
    num_tests = len(tests)
    avg_naive = {k: metrics_naive_total[k] / num_tests for k in metrics_naive_total}
    avg_adv = {k: metrics_adv_total[k] / num_tests for k in metrics_adv_total}
    avg_lex = {k: metrics_lex_total[k] / num_tests for k in metrics_lex_total}
    avg_mmr = {k: metrics_mmr_total[k] / num_tests for k in metrics_mmr_total}
    avg_sem = {k: metrics_semantic_total[k] / num_tests for k in metrics_semantic_total}
    avg_tfidf = {k: metrics_tfidf_total[k] / num_tests for k in metrics_tfidf_total}
    avg_bm25 = {k: metrics_bm25_total[k] / num_tests for k in metrics_bm25_total}
    avg_comp = {k: metrics_comp_total[k] / num_tests for k in metrics_comp_total}
    avg_multi = {k: metrics_multi_total[k] / num_tests for k in metrics_multi_total}
    avg_parent = {k: metrics_parent_total[k] / num_tests for k in metrics_parent_total}
    avg_ens = {k: metrics_ens_total[k] / num_tests for k in metrics_ens_total}

    print("\nAverages over all queries:")
    print(
        "Naïve average metrics (F/R/P/Rc): "
        f"{avg_naive['faithfulness']:.2f}/{avg_naive['relevance']:.2f}/{avg_naive['precision']:.2f}/{avg_naive['recall']:.2f}"
    )
    print(
        "Advanced average metrics (F/R/P/Rc): "
        f"{avg_adv['faithfulness']:.2f}/{avg_adv['relevance']:.2f}/{avg_adv['precision']:.2f}/{avg_adv['recall']:.2f}"
    )
    print(
        "Lexical average metrics (F/R/P/Rc): "
        f"{avg_lex['faithfulness']:.2f}/{avg_lex['relevance']:.2f}/{avg_lex['precision']:.2f}/{avg_lex['recall']:.2f}"
    )
    print(
        "MMR average metrics (F/R/P/Rc): "
        f"{avg_mmr['faithfulness']:.2f}/{avg_mmr['relevance']:.2f}/{avg_mmr['precision']:.2f}/{avg_mmr['recall']:.2f}"
    )

    print(
        "Semantic average metrics (F/R/P/Rc): "
        f"{avg_sem['faithfulness']:.2f}/{avg_sem['relevance']:.2f}/{avg_sem['precision']:.2f}/{avg_sem['recall']:.2f}"
    )
    print(
        "TF‑IDF average metrics (F/R/P/Rc): "
        f"{avg_tfidf['faithfulness']:.2f}/{avg_tfidf['relevance']:.2f}/{avg_tfidf['precision']:.2f}/{avg_tfidf['recall']:.2f}"
    )
    print(
        "BM25 average metrics (F/R/P/Rc): "
        f"{avg_bm25['faithfulness']:.2f}/{avg_bm25['relevance']:.2f}/{avg_bm25['precision']:.2f}/{avg_bm25['recall']:.2f}"
    )
    print(
        "Compression average metrics (F/R/P/Rc): "
        f"{avg_comp['faithfulness']:.2f}/{avg_comp['relevance']:.2f}/{avg_comp['precision']:.2f}/{avg_comp['recall']:.2f}"
    )
    print(
        "Multi‑Query average metrics (F/R/P/Rc): "
        f"{avg_multi['faithfulness']:.2f}/{avg_multi['relevance']:.2f}/{avg_multi['precision']:.2f}/{avg_multi['recall']:.2f}"
    )
    print(
        "Parent average metrics (F/R/P/Rc): "
        f"{avg_parent['faithfulness']:.2f}/{avg_parent['relevance']:.2f}/{avg_parent['precision']:.2f}/{avg_parent['recall']:.2f}"
    )
    print(
        "Ensemble average metrics (F/R/P/Rc): "
        f"{avg_ens['faithfulness']:.2f}/{avg_ens['relevance']:.2f}/{avg_ens['precision']:.2f}/{avg_ens['recall']:.2f}"
    )


if __name__ == "__main__":
    main_eval()
