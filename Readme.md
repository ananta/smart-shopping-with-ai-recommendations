# Smart Shopping AI - Certification Challenge Report

## Task 1: Defining your Problem and Audience

### Problem Description
**Problem**: Consumers struggle to find the best products that match their specific needs, budget, and preferences due to information overload and lack of personalized recommendations from trusted sources.

### Why This is a Problem
Consumers face significant challenges when making purchasing decisions. They often spend hours researching products across multiple websites, reading conflicting reviews, and comparing prices without having access to community-driven insights. The current e-commerce landscape provides generic recommendations that don't account for individual use cases, budgets, or specific requirements. Users need a solution that combines AI-powered analysis with community consensus to provide personalized, trustworthy product recommendations that save time and lead to better purchasing decisions.

**Target User**: Budget-conscious consumers who want to make informed purchasing decisions, particularly those researching specific product categories like home espresso equipment, headphones, laptops, or outdoor gear. These users value community feedback and want to ensure they're getting the best value for their money.

## Task 2: Propose a Solution

### Proposed Solution
Smart Shopping AI is an intelligent recommendation system that combines AI-powered goal analysis with community-driven product insights and real-time pricing data. The application allows users to describe their needs in natural language (e.g., "I want to set up a budget home espresso station under $500"), analyzes their requirements using LLM reasoning, retrieves relevant community recommendations, and provides real-time pricing and purchase links.

**User Experience**: Users interact with a large, intuitive text prompt where they describe their shopping goals. The system then provides a comprehensive analysis including categorized recommendations, community voting data, real-time pricing, and direct purchase links. Users can add products to a shopping cart and manage their selections, creating a seamless shopping experience that bridges the gap between research and purchase.

### Technology Stack

1. **LLM**: OpenAI GPT-4o - Chosen for its superior reasoning capabilities in analyzing user goals and synthesizing recommendations
2. **Embedding Model**: OpenAI text-embedding-3-small - Selected for semantic similarity matching between user queries and product descriptions
3. **Orchestration**: LangGraph - Used for structured workflow management and state transitions between analysis, retrieval, and synthesis
4. **Vector Database**: FAISS (in-memory) - Chosen for fast similarity search and easy deployment
5. **Evaluation**: Custom RAGAS like framework (had issues with ragas integration, so this one is pending) 
6. **User Interface**: React with TypeScript - Modern, responsive web interface with real-time feedback
7. **Serving & Inference**: FastAPI with Uvicorn - High-performance async API server

### Agentic Reasoning Implementation
The application uses a multi-agent approach with LangGraph:
- **Goal Analysis Agent**: Analyzes user input to extract relevant product categories
- **Retrieval Agent**: Implements multiple retrieval strategies (semantic, lexical, TF-IDF, BM25) to find relevant products
- **Synthesis Agent**: Combines recommendations with pricing data to generate natural language summaries
- **Pricing Agent**: Uses Tavily API to fetch real-time pricing and purchase links

## Task 3: Dealing with the Data

### Data Sources and External APIs

1. **Community Product Data** (`mock_data.json`): Curated dataset of product recommendations with community voting, including categories like espresso machines, grinders, headphones, backpacks, laptops, and coffee beans. Each entry contains product name, community comment, and vote count.

2. **Tavily Search API**: Used to fetch real-time pricing information and purchase links from major retailers (Amazon, Best Buy, Walmart, Target, Newegg). This provides current pricing data and direct purchase links.

3. **Reddit-style Data** (`reddit-data.json`): Additional community-driven product recommendations with timestamps and subreddit context.

### Chunking Strategy
**Default Strategy**: Semantic chunking with overlap. Products are chunked at the individual recommendation level, preserving the complete context of each product recommendation including the community comment, vote count, and category. This approach maintains the integrity of community feedback while enabling semantic search.

**Rationale**: This chunking strategy preserves the full context of each recommendation, allowing the system to understand not just what product is recommended, but why it's recommended and how the community feels about it. The overlap ensures that related concepts aren't split across chunks.

### Additional Data Requirements
- **User interaction logs**: To improve recommendations based on user behavior
- **Price history data**: To track price trends and provide better value recommendations
- **Product specifications**: For more detailed filtering and comparison capabilities

## Task 4: Building a Quick End-to-End Agentic RAG Prototype

### Implementation Details

The prototype has been successfully built and deployed as a local endpoint with the following architecture:

**Backend (FastAPI)**:
- Endpoint: `http://localhost:8000/recommend`
- LangGraph workflow with three main nodes: goal analysis, retrieval, synthesis
- Multiple retrieval strategies (naive, advanced, semantic, lexical, TF-IDF, BM25, MMR, ensemble)
- Tavily API integration for real-time pricing and purchase links

**Frontend (React)**:
- Modern UI with large prompt input area
- Real-time recommendations display with pricing and purchase links
- Shopping cart functionality
- Responsive design for all devices

**Key Features Implemented**:
- Natural language goal analysis
- Community-driven product recommendations
- Real-time pricing via Tavily API
- Direct purchase links to major retailers
- Shopping cart management
- Multiple retrieval strategies
- Error handling and loading states

## Task 5: Creating a Golden Test Data Set

### Performance Analysis

**Key Insights**:
- Community voting data provides strong grounding for recommendations
- Room for improvement in retrieval precision through advanced techniques
- Real-time pricing integration enhances the user experience significantly

## Task 6: The Benefits of Advanced Retrieval

### Advanced Retrieval Techniques Implemented

1. **Semantic Retrieval**: Uses OpenAI embeddings for semantic similarity matching. Useful for capturing meaning beyond exact keyword matches.

2. **Lexical Retrieval**: Implements keyword overlap scoring. Useful for specific product names and technical terms.

3. **TF-IDF Retrieval**: Uses TF-IDF vectorization with cosine similarity. Useful for balancing term frequency with document importance.

4. **BM25 Retrieval**: Implements BM25 ranking algorithm. Useful for better handling of document length and term frequency.

5. **MMR (Maximal Marginal Relevance)**: Balances relevance with diversity. Useful for providing varied recommendations.

6. **Ensemble Retrieval**: Combines lexical and TF-IDF scores. Useful for robust performance across different query types.

7. **Multi-Query Retrieval**: Uses query expansion with synonyms. Useful for handling different ways users express the same need.

8. **Parent Document Retrieval**: Aggregates category-level documents. Useful for broad category matching.

### Testing Results

Each retrieval technique was tested with sample queries:
- "I want to set up a budget home espresso station under $500"
- "Looking for noise-cancelling headphones for travel under $200"
- "Need an ultralight backpack for thru-hiking"

**Performance Comparison**:
- **Semantic Retrieval**: Best for understanding user intent and context
- **Lexical Retrieval**: Best for exact product name matching
- **Ensemble Retrieval**: Most robust across different query types
- **MMR Retrieval**: Best for providing diverse recommendations

## Task 7: Assessing Performance

I didn't use RAGAS for this one because our product recommendation app relies heavily on Reddit-based community wisdon, where the value lies not jsut in factual correctness but in user concsensus, voting patterns and contexttual nuance.

But, in future, I'll replace my custom evaluator with ragas for syntehtic test set generation and baseline comparision for quantitivie scoring.

### Performance Comparison Results
| Retriever           | Faithfulness (F) | Response Relevance (R) | Context Precision (P) | Context Recall (Rc) |
|---------------------|------------------|--------------------------|------------------------|----------------------|
| Naïve               | 0.67             | 0.33                     | 0.05                   | 0.17                 |
| Advanced            | 0.67             | 0.33                     | 0.05                   | 0.17                 |
| Lexical             | 0.67             | 0.00                     | 0.09                   | 0.00                 |
| MMR                 | 0.67             | 0.00                     | 0.09                   | 0.00                 |
| Semantic            | 0.67             | 0.00                     | 0.06                   | 0.00                 |
| TF‑IDF              | 0.67             | 0.33                     | 0.08                   | 0.17                 |
| BM25                | 0.67             | 0.00                     | 0.09                   | 0.00                 |
| Compression         | 0.67             | 0.33                     | 0.06                   | 0.04                 |
| Multi‑Query         | 0.67             | 0.00                     | 0.09                   | 0.00                 |
| Parent              | 0.67             | 0.33                     | 0.05                   | 0.17                 |
| Ensemble            | 0.67             | 0.00                     | 0.10                   | 0.00                 |


### Performance Improvements

| Metric                     | **Parent Retriever** | **Semantic Retriever** | **Improvement** |
| -------------------------- | -------------------- | ---------------------- | --------------- |
| **Faithfulness (F)**       | 0.67                 | 0.67                   | 0%              |
| **Response Relevance (R)** | 0.33                 | 0.00                   | **+33%**        |
| **Context Precision (P)**  | 0.05                 | 0.06                   | -0.01 (-1%)     |
| **Context Recall (Rc)**    | 0.17                 | 0.00                   | **+17%**        |

- ✅ Parent Retriever is clearly superior for Reddit-style community recommendations.
- ❌ Semantic Retriever fails to retrieve relevant or complete context in this use case.
- 🔍 Despite slightly better precision, Semantic Retriever misses critical recall and relevance, making it unsuitable for your product-focused application.


### Planned Improvements for Second Half

The Smart Shopping AI system demonstrates the power of combining AI reasoning with community insights and real-time data to create a comprehensive shopping recommendation platform that addresses real user needs. 

* Integrate exteranl evaluator like RAGAS for benchmarking, synthetic dataset generation and reporting
* Deploy a prod ready app: [dartpick.com](dartpick.com)