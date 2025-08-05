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
import requests
from typing import TypedDict, List, Dict, Any
import re

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
    recommendations: Dict[str, str]
    final_output: str


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
        idf[w] = max(0.0, ( (N - count + 0.5) / (count + 0.5) ))
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
# Tavily API integration for product pricing and purchase links
# -----------------------------------------------------------------------------

def get_product_pricing_and_links(product_name: str, category: str) -> Dict[str, Any]:
    """Fetch product pricing and purchase links using Tavily API.
    
    Args:
        product_name: Name of the product to search for
        category: Product category for context
        
    Returns:
        Dictionary containing pricing info and purchase links
    """
    if not TAVILY_API_KEY:
        return {
            "price": None,
            "purchase_link": None,
            "available_stores": [],
            "error": "Tavily API key not configured"
        }
    
    try:
        # Search query for the product
        search_query = f"{product_name} {category} price buy online"
        
        # Tavily search API call
        url = "https://api.tavily.com/search"
        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "content-type": "application/json"
        }
        
        payload = {
            "query": search_query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 5
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract pricing and purchase information
        price_info = None
        purchase_link = None
        available_stores = []
        
        # Look for pricing information in the results
        for result in data.get("results", []):
            content = result.get("content", "").lower()
            url = result.get("url", "")
            
            # Look for price patterns
            import re
            price_patterns = [
                r'\$[\d,]+\.?\d*',
                r'[\d,]+\.?\d*\s*(?:usd|dollars?)',
                r'price[:\s]*\$?[\d,]+\.?\d*'
            ]
            
            for pattern in price_patterns:
                price_match = re.search(pattern, content)
                if price_match and not price_info:
                    price_info = price_match.group()
                    break
            
            # Look for purchase links (Amazon, Best Buy, etc.)
            if any(store in url.lower() for store in ['amazon', 'bestbuy', 'walmart', 'target', 'newegg']):
                if not purchase_link:
                    purchase_link = url
                available_stores.append({
                    "name": result.get("title", "Unknown Store"),
                    "url": url
                })
        
        # If no specific price found, try to extract from answer
        if not price_info and data.get("answer"):
            answer = data.get("answer", "").lower()
            price_match = re.search(r'\$[\d,]+\.?\d*', answer)
            if price_match:
                price_info = price_match.group()
        
        return {
            "price": price_info,
            "purchase_link": purchase_link,
            "available_stores": available_stores[:3],  # Limit to top 3 stores
            "search_results": len(data.get("results", [])),
            "error": None
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "price": None,
            "purchase_link": None,
            "available_stores": [],
            "error": f"API request failed: {str(e)}"
        }
    except Exception as e:
        return {
            "price": None,
            "purchase_link": None,
            "available_stores": [],
            "error": f"Unexpected error: {str(e)}"
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
    """Use the LLM to craft the final recommendation summary.

    This node takes the recommendations dictionary and constructs a single
    string with each category and its associated recommendation on its own
    line.  It then feeds that into the LLM to generate a concise
    explanation of why the recommended products meet the user's needs.  The
    result is stored under `final_output`.
    """
    # Build a human‑readable string from the structured recommendations.  Each
    # line contains the category name followed by the recommended product and
    # a brief rationale.  If no product was found, include the fallback
    # message.  This string is used to prompt the LLM for a friendly
    # summary.
    lines = []
    for cat, info in state.get("recommendations", {}).items():
        # Handle the advanced retriever output (list of top comments)
        if isinstance(info, dict) and "top_comments" in info:
            comments_list = info.get("top_comments", [])
            if comments_list:
                # Join multiple comments into one line separated by semicolons
                comment_strs = []
                for c in comments_list:
                    comment_strs.append(
                        f"{c['product']} (votes: {c['votes']}) – {c['comment']}"
                    )
                lines.append(f"{cat}: " + "; ".join(comment_strs))
            else:
                # Fallback message if no comments are found
                lines.append(f"{cat}: {info.get('message', 'No recommendations found.')}")
        # Handle the naïve retriever output (single comment)
        elif isinstance(info, dict) and info.get("product"):
            lines.append(
                f"{cat}: {info['product']} (votes: {info['votes']}) – {info['comment']}"
            )
        else:
            # Use the custom message if provided, otherwise fall back to a generic notice
            message = info.get("message") if isinstance(info, dict) else str(info)
            lines.append(f"{cat}: {message}")
    recommendations_str = "\n".join(lines)
    chain = synth_prompt | llm
    response = chain.invoke({
        "user_goal": state["user_goal"],
        "recommendations": recommendations_str,
    })
    return {
        **state,
        "final_output": response.content,
    }


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

# Define the order of execution: analysis → retrieval → synthesis
builder.set_entry_point("goal_analyzer")
builder.add_edge("goal_analyzer", "retriever")
builder.add_edge("retriever", "synthesizer")
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
    allow_origins=["http://localhost:3000"],
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
    }
    result_state = graph.invoke(initial_state)
    return {
        "user_goal": req.user_goal,
        "categories": result_state.get("category_plan", []),
        "recommendations": result_state.get("recommendations", {}),
        "summary": result_state.get("final_output", ""),
    }


def run_demo(user_goal: str) -> str:
    """Invoke the graph for a given user goal and return the final summary."""
    initial_state: ToolkitState = {
        "user_goal": user_goal,
        "category_plan": [],
        "recommendations": {},
        "final_output": "",
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
