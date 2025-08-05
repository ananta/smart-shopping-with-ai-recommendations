import React from 'react';
import { Product, RecommendationResponse } from '../App';

interface RecommendationsProps {
  recommendations: RecommendationResponse;
  onAddToCart: (product: Product) => void;
}

const Recommendations: React.FC<RecommendationsProps> = ({ recommendations, onAddToCart }) => {
  const extractProductsFromRecommendations = (): Product[] => {
    const products: Product[] = [];
    
    Object.entries(recommendations.recommendations).forEach(([category, data]) => {
      if (data && typeof data === 'object') {
        // Handle different response formats
        if (data.top_comments && Array.isArray(data.top_comments)) {
          data.top_comments.forEach((comment: any, index: number) => {
            if (comment.product) {
              products.push({
                id: `${category}-${index}`,
                name: comment.product,
                category: category,
                description: comment.comment || 'No description available',
                votes: comment.votes || 0,
              });
            }
          });
        } else if (data.product) {
          // Handle single product format
          products.push({
            id: `${category}-0`,
            name: data.product,
            category: category,
            description: data.comment || 'No description available',
            votes: data.votes || 0,
          });
        }
      }
    });
    
    return products;
  };

  const products = extractProductsFromRecommendations();

  return (
    <div className="recommendations">
      <h2>AI Recommendations</h2>
      
      <div className="summary-section">
        <h3>Summary</h3>
        <p className="summary-text">{recommendations.summary}</p>
      </div>

      {products.length > 0 ? (
        <div className="products-grid">
          <h3>Recommended Products</h3>
          <div className="products-list">
            {products.map((product) => (
              <div key={product.id} className="product-card">
                <div className="product-header">
                  <h4 className="product-name">{product.name}</h4>
                  <span className="product-category">{product.category}</span>
                </div>
                
                <p className="product-description">{product.description}</p>
                
                {product.votes && (
                  <div className="product-votes">
                    <span className="votes-badge">👍 {product.votes} votes</span>
                  </div>
                )}
                
                <button
                  className="add-to-cart-button"
                  onClick={() => onAddToCart(product)}
                >
                  Add to Cart
                </button>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="no-recommendations">
          <p>No specific products found, but here's what we found:</p>
          <div className="categories-found">
            <h3>Categories identified:</h3>
            <ul>
              {recommendations.categories.map((category, index) => (
                <li key={index}>{category}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default Recommendations; 