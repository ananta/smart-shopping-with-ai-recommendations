import React, { useState } from 'react';
import './App.css';
import ProductPrompt from './components/ProductPrompt';
import Recommendations from './components/Recommendations';
import Cart from './components/Cart';

export interface Product {
  id: string;
  name: string;
  category: string;
  description: string;
  votes?: number;
  price?: string;
  purchaseLink?: string;
  availableStores?: Array<{
    name: string;
    url: string;
  }>;
}

export interface RecommendationResponse {
  user_goal: string;
  categories: string[];
  recommendations: Record<string, any>;
  summary: string;
}

function App() {
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cart, setCart] = useState<Product[]>([]);

  const handleGetRecommendations = async (userGoal: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_goal: userGoal }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: RecommendationResponse = await response.json();
      setRecommendations(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const addToCart = (product: Product) => {
    setCart(prevCart => {
      const existingProduct = prevCart.find(item => item.id === product.id);
      if (existingProduct) {
        return prevCart; // Product already in cart
      }
      return [...prevCart, product];
    });
  };

  const removeFromCart = (productId: string) => {
    setCart(prevCart => prevCart.filter(item => item.id !== productId));
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Smart Shopping AI</h1>
        <p>Get personalized product recommendations powered by AI</p>
      </header>
      
      <main className="App-main">
        <div className="container">
          <ProductPrompt 
            onGetRecommendations={handleGetRecommendations}
            loading={loading}
          />
          
          {error && (
            <div className="error-message">
              <p>Error: {error}</p>
              <p>Make sure your backend server is running on http://localhost:8000</p>
            </div>
          )}
          
          {recommendations && (
            <Recommendations 
              recommendations={recommendations}
              onAddToCart={addToCart}
            />
          )}
        </div>
        
        <Cart 
          items={cart}
          onRemoveItem={removeFromCart}
        />
      </main>
    </div>
  );
}

export default App;
