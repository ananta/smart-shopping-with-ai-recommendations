# Smart Shopping AI with AI Recommendations

A full-stack application that provides intelligent product recommendations based on user goals and community feedback. The system uses AI to analyze user requirements and suggest the best products from curated community data.

## 🚀 Features

- **AI-Powered Recommendations**: Uses LangChain and OpenAI to analyze user goals and provide personalized product suggestions
- **Community-Driven Data**: Recommendations based on real community feedback and voting
- **Modern React Frontend**: Beautiful, responsive UI with shopping cart functionality
- **FastAPI Backend**: Robust API with multiple retrieval strategies
- **Multiple Retrieval Methods**: Supports various AI retrieval techniques (semantic, lexical, TF-IDF, BM25, etc.)

## 🏗️ Architecture

```
smart-shopping-with-ai-recommendations/
├── main.py                 # FastAPI backend server
├── mock_data.json          # Sample product data
├── requirements.txt        # Python dependencies
├── start-app.sh           # Quick start script
├── app/                   # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── App.tsx       # Main app component
│   │   └── App.css       # Modern styling
│   └── package.json       # Node.js dependencies
└── data/                  # Data files
    └── reddit-data.json   # Additional sample data
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **LangChain**: AI/LLM orchestration
- **OpenAI**: GPT-4 for goal analysis and synthesis
- **scikit-learn**: TF-IDF and similarity calculations
- **Uvicorn**: ASGI server

### Frontend
- **React 18**: Modern React with TypeScript
- **Modern CSS**: Flexbox, Grid, animations
- **Fetch API**: Backend communication

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API key
- Tavily API key (for product pricing and purchase links)

### Quick Start

1. **Clone and setup**:
```bash
git clone <repository-url>
cd smart-shopping-with-ai-recommendations
```

2. **Set up environment variables**:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

3. **Run the application**:
```bash
./start-app.sh
```

This will:
- Install Python dependencies
- Install React dependencies
- Start the backend server on http://localhost:8000
- Start the React app on http://localhost:3000

### Manual Setup

#### Backend Setup
```bash
pip install -r requirements.txt
python main.py
```

#### Frontend Setup
```bash
cd app
npm install
npm start
```

## 🎯 Usage

### Using the Web Interface

1. **Open the app**: Navigate to http://localhost:3000
2. **Enter your goal**: Use the large text area to describe what you're looking for
3. **Get recommendations**: Click "Get Recommendations" to receive AI-powered suggestions
4. **Add to cart**: Click "Add to Cart" on products you're interested in
5. **Manage cart**: View and manage your selected items in the cart sidebar

### Example Prompts

- "I want to set up a budget home espresso station under $500"
- "Looking for noise-cancelling headphones for travel under $200"
- "Need an ultralight backpack for thru-hiking"
- "Want a reliable laptop for programming and gaming"

### API Usage

The backend provides a REST API at `http://localhost:8000/recommend`:

**Features:**
- AI-powered product recommendations based on user goals
- Real-time pricing information via Tavily API
- Purchase links to major retailers (Amazon, Best Buy, Walmart, etc.)
- Community voting data from Reddit-style recommendations

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_goal": "I want to set up a budget home espresso station under $500"}'
```

Response:
```json
{
  "user_goal": "I want to set up a budget home espresso station under $500",
  "categories": ["espresso machine", "grinder"],
  "recommendations": {
    "espresso machine": {
      "top_comments": [
        {
          "product": "Gaggia Classic Pro",
          "comment": "Best entry-level espresso machine under $500...",
          "votes": 320,
          "pricing": {
            "price": "$399.99",
            "purchase_link": "https://amazon.com/...",
            "available_stores": [
              {"name": "Amazon", "url": "https://amazon.com/..."},
              {"name": "Best Buy", "url": "https://bestbuy.com/..."}
            ]
          }
        }
      ]
    }
  },
  "summary": "Based on your goal of setting up a budget home espresso station..."
}
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `DATA_FILE`: Path to data file (default: `mock_data.json`)
- `RETRIEVER_STRATEGY`: Retrieval method to use (default: `naive`)

### Retrieval Strategies

The backend supports multiple retrieval strategies:

- `naive`: Simple top-vote selection
- `advanced`: Top-2 vote-based retrieval
- `lexical`: Lexical overlap scoring
- `semantic`: OpenAI embedding similarity
- `tfidf`: TF-IDF cosine similarity
- `bm25`: BM25 ranking
- `mmr`: Maximal Marginal Relevance
- `ensemble`: Combined lexical + TF-IDF

Set the strategy:
```bash
export RETRIEVER_STRATEGY="semantic"
```

## 📊 Data Structure

The system uses JSON data with the following structure:

```json
[
  {
    "category": "espresso machine",
    "context": [
      {
        "product": "Gaggia Classic Pro",
        "comment": "Best entry-level espresso machine...",
        "votes": 320
      }
    ]
  }
]
```

## 🎨 Frontend Components

### ProductPrompt
- Large text area for user input
- Example prompts for guidance
- Loading states and validation

### Recommendations
- Displays AI recommendations
- Product cards with voting information
- Add to cart functionality

### Cart
- Shopping cart management
- Remove items functionality
- Clear cart option

## 🔍 Advanced Features

### Multiple Retrieval Methods
The backend implements various information retrieval techniques:

1. **Simple Reddit Retriever**: Basic vote-based selection
2. **Lexical Retriever**: Keyword overlap scoring
3. **Semantic Retriever**: OpenAI embedding similarity
4. **TF-IDF Retriever**: Vector-based similarity
5. **BM25 Retriever**: Advanced ranking algorithm
6. **MMR Retriever**: Diversity-aware selection
7. **Ensemble Retriever**: Combined scoring methods

### LangGraph Workflow
The system uses LangGraph for structured AI workflows:

1. **Goal Analysis**: Extract shopping categories from user input
2. **Retrieval**: Find relevant products using selected strategy
3. **Synthesis**: Generate natural language recommendations

## 🚀 Deployment

### Backend Deployment
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Deployment
```bash
cd app
npm run build
# Deploy the build/ folder to your hosting service
```

## 🐛 Troubleshooting

### Common Issues

1. **CORS Errors**: Backend includes CORS middleware for localhost:3000
2. **API Key Issues**: Ensure OPENAI_API_KEY is set correctly
3. **Port Conflicts**: Backend runs on 8000, frontend on 3000
4. **Data File**: Ensure mock_data.json exists in project root

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- LangChain for AI orchestration
- FastAPI for the backend framework
- React team for the frontend framework
