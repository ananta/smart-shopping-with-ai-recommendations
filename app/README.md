# Smart Shopping AI - React Frontend

A modern React application that provides an intuitive interface for the Smart Shopping AI recommendation system.

## Features

- **Large Prompt Input**: Users can describe their shopping needs in natural language
- **AI-Powered Recommendations**: Integrates with the backend API to get personalized product recommendations
- **Shopping Cart**: Add recommended products to cart and manage your selections
- **Modern UI**: Beautiful, responsive design with smooth animations
- **Real-time Feedback**: Loading states and error handling for better UX

## Prerequisites

Make sure you have the backend server running. The React app expects the API to be available at `http://localhost:8000`.

## Installation

1. Install dependencies:
```bash
npm install
```

## Running the Application

1. Start the development server:
```bash
npm start
```

2. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

## Usage

1. **Enter Your Shopping Goal**: Use the large text area to describe what you're looking for. Be specific about your needs, budget, and preferences.

2. **Get Recommendations**: Click "Get Recommendations" to send your request to the AI backend.

3. **Review Results**: The AI will provide:
   - A summary of recommendations
   - Specific product suggestions with community votes
   - Categories that match your needs

4. **Add to Cart**: Click "Add to Cart" on any product you're interested in.

5. **Manage Your Cart**: View your selected items in the cart sidebar and remove items as needed.

## Example Prompts

- "I want to set up a budget home espresso station under $500"
- "Looking for noise-cancelling headphones for travel under $200"
- "Need an ultralight backpack for thru-hiking"
- "Want a reliable laptop for programming and gaming"

## API Integration

The app communicates with the backend API at `http://localhost:8000/recommend`:

- **Method**: POST
- **Content-Type**: application/json
- **Body**: `{ "user_goal": "your shopping description" }`
- **Response**: JSON with categories, recommendations, and summary

## Development

### Project Structure

```
src/
├── components/
│   ├── ProductPrompt.tsx    # Large input area for user goals
│   ├── Recommendations.tsx  # Displays AI recommendations
│   └── Cart.tsx            # Shopping cart management
├── App.tsx                 # Main application component
└── App.css                 # Modern styling
```

### Key Components

- **ProductPrompt**: Handles user input and API calls
- **Recommendations**: Displays AI recommendations with add-to-cart functionality
- **Cart**: Manages selected products with remove/clear options

### Styling

The app uses modern CSS with:
- Gradient backgrounds
- Card-based layouts
- Smooth animations and transitions
- Responsive design for mobile devices
- Glassmorphism effects

## Troubleshooting

### Backend Connection Issues

If you see connection errors:

1. Make sure the backend server is running on `http://localhost:8000`
2. Check that the API endpoint `/recommend` is available
3. Verify your backend is configured to accept CORS requests from `http://localhost:3000`

### Common Issues

- **CORS Errors**: The backend needs to allow requests from the React app
- **API Timeout**: Large requests might take time; the UI shows loading states
- **No Recommendations**: Try being more specific in your product description

## Building for Production

To create a production build:

```bash
npm run build
```

This creates an optimized build in the `build` folder that can be deployed to any static hosting service.

## Technologies Used

- **React 18** with TypeScript
- **Modern CSS** with Flexbox and Grid
- **Fetch API** for backend communication
- **Responsive Design** for all screen sizes
