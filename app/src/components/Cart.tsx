import React from 'react';
import { Product } from '../App';

interface CartProps {
  items: Product[];
  onRemoveItem: (productId: string) => void;
}

const Cart: React.FC<CartProps> = ({ items, onRemoveItem }) => {
  if (items.length === 0) {
    return (
      <div className="cart cart-empty">
        <h3>Shopping Cart</h3>
        <p>Your cart is empty. Add some products from the recommendations!</p>
      </div>
    );
  }

  return (
    <div className="cart">
      <h3>Shopping Cart ({items.length} items)</h3>
      
      <div className="cart-items">
        {items.map((item) => (
          <div key={item.id} className="cart-item">
            <div className="cart-item-content">
              <h4 className="cart-item-name">{item.name}</h4>
              <p className="cart-item-category">{item.category}</p>
              <p className="cart-item-description">{item.description}</p>
              
              <div className="cart-item-info">
                {item.votes && (
                  <span className="cart-item-votes">👍 {item.votes} votes</span>
                )}
                {item.price && (
                  <span className="cart-item-price">💰 {item.price}</span>
                )}
              </div>
              
              {item.purchaseLink && (
                <a 
                  href={item.purchaseLink} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="cart-purchase-link"
                >
                  🛒 Buy Now
                </a>
              )}
            </div>
            
            <button
              className="remove-from-cart-button"
              onClick={() => onRemoveItem(item.id)}
              aria-label={`Remove ${item.name} from cart`}
            >
              Remove
            </button>
          </div>
        ))}
      </div>
      
      <div className="cart-actions">
        <button className="checkout-button" disabled>
          Checkout (Coming Soon)
        </button>
        <button 
          className="clear-cart-button"
          onClick={() => items.forEach(item => onRemoveItem(item.id))}
        >
          Clear Cart
        </button>
      </div>
    </div>
  );
};

export default Cart; 