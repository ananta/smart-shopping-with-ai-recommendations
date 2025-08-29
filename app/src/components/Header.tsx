import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ShoppingCart, Search, Menu, X, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import Logo from "../assets/logo.svg";

interface HeaderProps {
  cartItems: number;
  onCartClick: () => void;
}

export const Header = ({ cartItems, onCartClick }: HeaderProps) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  return (
    <header className="sticky top-0 z-50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-2">
            <img
              src={Logo}
              className="w-10 h-10 rounded-full"
              alt="Dartpick Logo"
            />
            <span className="text-xl font-bold text-primary">DartPick</span>
          </div>

          {/* Desktop Search */}
          {/* <div className="hidden md:flex flex-1 max-w-md mx-8"> */}
          {/*   <div className="relative w-full"> */}
          {/*     <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" /> */}
          {/*     <Input */}
          {/*       placeholder="Search for products..." */}
          {/*       value={searchQuery} */}
          {/*       onChange={e => setSearchQuery(e.target.value)} */}
          {/*       className="pl-10 pr-4" */}
          {/*     /> */}
          {/*   </div> */}
          {/* </div> */}

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            {/* <Button variant="ghost" className="text-sm"> */}
            {/*   Browse */}
            {/* </Button> */}
            {/* <Button variant="ghost" className="text-sm"> */}
            {/*   AI Picks */}
            {/* </Button> */}
            {/* <Button variant="ghost" className="text-sm"> */}
            {/*   Community */}
            {/* </Button> */}
            {/* <Button variant="outline" className="text-sm"> */}
            {/*   Sign In */}
            {/* </Button> */}
            <Button variant="cart" onClick={onCartClick} className="relative">
              <ShoppingCart className="h-4 w-4" />
              Cart
              {cartItems > 0 && (
                <Badge className="absolute -top-2 -right-2 h-5 w-5 p-0 flex items-center justify-center bg-primary text-primary-foreground">
                  {cartItems}
                </Badge>
              )}
            </Button>
          </nav>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? (
              <X className="h-5 w-5" />
            ) : (
              <Menu className="h-5 w-5" />
            )}
          </Button>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="md:hidden py-4 border-t">
            <div className="space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
                <Input
                  placeholder="Search for products..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="pl-10 pr-4"
                />
              </div>
              <div className="space-y-2">
                {/* <Button variant="ghost" className="w-full justify-start"> */}
                {/*   Browse */}
                {/* </Button> */}
                {/* <Button variant="ghost" className="w-full justify-start"> */}
                {/*   AI Picks */}
                {/* </Button> */}
                {/* <Button variant="ghost" className="w-full justify-start"> */}
                {/*   Community */}
                {/* </Button> */}
                {/* <Button variant="outline" className="w-full"> */}
                {/*   Sign In */}
                {/* </Button> */}
                <Button
                  variant="cart"
                  onClick={onCartClick}
                  className="w-full relative"
                >
                  <ShoppingCart className="h-4 w-4" />
                  Cart ({cartItems})
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};
