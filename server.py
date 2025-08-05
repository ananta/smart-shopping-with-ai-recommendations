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
from typing import TypedDict, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Data loading and indexing
# -----------------------------------------------------------------------------

# The mock data describes three typical community recommendation scenarios.
# Each entry contains a question, a list of comment objects, a list of
# categories associated with the thread, and some metadata.  We load this
# file at startup and build a simple index mapping categories to comment
# suggestions.  When you swap in real Reddit data later, ensure it
# conforms to the same general structure.

load_dotenv()
DATA_FILE = "data/mock_data.json"

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

# Prompt for the goal analysis step.  It instructs the LLM to return a
# simple list of shopping categories relevant to the user goal.  We avoid
# bullet characters so that we can easily split the response by newlines.
analyze_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant that maps user goals to one or more of the following categories: "
        "Espresso Machines, Headphones, Hiking Gear. "
        "Read the user's goal and return the category names that best match. "
        "Output one category per line with no bullets or extra commentary."
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

def simple_reddit_retriever(state: ToolkitState) -> ToolkitState:
    """Retrieve the best community product suggestions for each category.

    For every category found by the analysis step, look up all associated
    comments in the `category_index`.  Select the comment with the highest
    vote count and build a recommendation string that includes the
    product name, the comment text and the vote total.  If no comments
    are found for a category, insert a fallback message.
    """
    updated_recs = state.get("recommendations", {}).copy()
    for category in state.get("category_plan", []):
        key = category.lower()
        comments = category_index.get(key, [])
        if comments:
            # Select the comment with the highest vote count
            best_comment = max(comments, key=lambda c: c.get("votes", 0))
            updated_recs[category] = (
                f"{best_comment['product']} – {best_comment['comment']} "
                f"(votes: {best_comment['votes']})"
            )
        else:
            updated_recs[category] = f"No strong Reddit consensus found for {category}"
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
    """Use the LLM to craft the final recommendation summary.

    This node takes the recommendations dictionary and constructs a single
    string with each category and its associated recommendation on its own
    line.  It then feeds that into the LLM to generate a concise
    explanation of why the recommended products meet the user's needs.  The
    result is stored under `final_output`.
    """
    recommendations_str = "\n".join(
        f"{cat}: {desc}" for cat, desc in state.get("recommendations", {}).items()
    )
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
from pydantic import BaseModel

# Instantiate the FastAPI application
app = FastAPI()

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
