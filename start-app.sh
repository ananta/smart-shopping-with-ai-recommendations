#!/bin/bash

echo "🚀 Starting Smart Shopping AI Application"
echo "=========================================="

# Check for required API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set. Backend will not work without it."
    echo "   Set it with: export OPENAI_API_KEY='your-key-here'"
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "⚠️  Warning: TAVILY_API_KEY not set. Product pricing and purchase links will not be available."
    echo "   Set it with: export TAVILY_API_KEY='your-key-here'"
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if the required files exist
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Make sure you're in the project root directory."
    exit 1
fi

if [ ! -d "app" ]; then
    echo "❌ app directory not found. Make sure the React app is in the app/ directory."
    exit 1
fi

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt 2>/dev/null || echo "⚠️  No requirements.txt found, skipping Python dependencies"

echo "📦 Installing React dependencies..."
cd app
npm install
cd ..

echo ""
echo "🌐 Starting backend server..."
echo "Backend will be available at: http://localhost:8000"
echo "API endpoint: http://localhost:8000/recommend"
echo ""

# Start backend in background
python3 main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

echo "🎨 Starting React development server..."
echo "Frontend will be available at: http://localhost:3000"
echo ""

# Start frontend in background
cd app
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both servers are starting up!"
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:8000"
echo ""
echo "💡 Try this example prompt:"
echo "   'I want to set up a budget home espresso station under $500'"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait 