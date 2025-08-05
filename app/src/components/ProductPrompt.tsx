import React, { useState } from 'react';

interface ProductPromptProps {
  onGetRecommendations: (userGoal: string) => void;
  loading: boolean;
}

const ProductPrompt: React.FC<ProductPromptProps> = ({ onGetRecommendations, loading }) => {
  const [userGoal, setUserGoal] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (userGoal.trim()) {
      onGetRecommendations(userGoal.trim());
    }
  };

  return (
    <div className="product-prompt">
      <h2>What are you looking for?</h2>
      <p className="prompt-description">
        Describe what you want to buy, your budget, preferences, or any specific requirements. 
        Our AI will analyze your needs and recommend the best products based on community feedback.
      </p>
      
      <form onSubmit={handleSubmit} className="prompt-form">
        <textarea
          value={userGoal}
          onChange={(e) => setUserGoal(e.target.value)}
          placeholder="e.g., I want to set up a budget home espresso station under $500, looking for reliable equipment that makes good coffee..."
          className="prompt-textarea"
          rows={6}
          disabled={loading}
        />
        
        <button 
          type="submit" 
          className="submit-button"
          disabled={loading || !userGoal.trim()}
        >
          {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
        </button>
      </form>
      
      <div className="example-prompts">
        <h3>Example prompts:</h3>
        <ul>
          <li>"I need noise-cancelling headphones for travel under $200"</li>
          <li>"Looking for an ultralight backpack for thru-hiking"</li>
          <li>"Want to build a home espresso setup on a budget"</li>
          <li>"Need a reliable laptop for programming and gaming"</li>
        </ul>
      </div>
    </div>
  );
};

export default ProductPrompt; 