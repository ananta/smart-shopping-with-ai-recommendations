import { useState } from "react";
import { Header } from "@/components/Header";
import { Hero } from "@/components/Hero";
import { Features } from "@/components/Features";
import { ProductGrid } from "@/components/ProductGrid";
import { Cart } from "@/components/Cart";
import { Footer } from "@/components/Footer";
import { AIRecommendations } from "@/components/AIRecommendations";

interface CartItem {
  id: number;
  name: string;
  price: number;
  quantity: number;
  image: string;
  aiRecommended?: boolean;
}

interface Product {
  id: number;
  name: string;
  price: number;
  originalPrice?: number;
  rating: number;
  reviews: number;
  image: string;
  category: string;
  aiRecommended?: boolean;
  trending?: boolean;
}

const Index = () => {
  const [cartItems, setCartItems] = useState<CartItem[]>([]);
  const [isCartOpen, setIsCartOpen] = useState(false);
  const [isAIRecommendationsOpen, setIsAIRecommendationsOpen] = useState(false);

  const addToCart = (product: Product) => {
    setCartItems(prev => {
      const existingItem = prev.find(item => item.id === product.id);
      if (existingItem) {
        return prev.map(item =>
          item.id === product.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        );
      }
      return [
        ...prev,
        {
          id: product.id,
          name: product.name,
          price: product.price,
          quantity: 1,
          image: product.image,
          aiRecommended: product.aiRecommended
        }
      ];
    });
  };

  const updateQuantity = (id: number, quantity: number) => {
    setCartItems(prev =>
      prev.map(item => (item.id === id ? { ...item, quantity } : item))
    );
  };

  const removeItem = (id: number) => {
    setCartItems(prev => prev.filter(item => item.id !== id));
  };

  const totalItems = cartItems.reduce((sum, item) => sum + item.quantity, 0);

  return (
    <div className="min-h-screen bg-background">
      <Header cartItems={totalItems} onCartClick={() => setIsCartOpen(true)} />
      <Hero onGetAIPicks={() => setIsAIRecommendationsOpen(true)} />
      <Features />
      {/* <ProductGrid onAddToCart={addToCart} /> */}
      <Footer />

      <Cart
        items={cartItems}
        onUpdateQuantity={updateQuantity}
        onRemoveItem={removeItem}
        isOpen={isCartOpen}
        onClose={() => setIsCartOpen(false)}
      />

      <AIRecommendations
        isOpen={isAIRecommendationsOpen}
        onClose={() => setIsAIRecommendationsOpen(false)}
      />
    </div>
  );
};

export default Index;
