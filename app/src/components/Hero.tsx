import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search, Sparkles, Users, Brain } from "lucide-react";
import heroPhone from "@/assets/hero-phone.jpg";
import businessmen from "../assets/buesiness.json";
import { Player } from "@lottiefiles/react-lottie-player";

interface HeroProps {
  onGetAIPicks: () => void;
}

export const Hero = ({ onGetAIPicks }: HeroProps) => {
  return (
    <section className="relative min-h-[80vh] flex items-center justify-center bg-background overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-20 left-10 w-20 h-20 bg-primary/5 rounded-full blur-xl" />
      <div className="absolute bottom-20 right-10 w-32 h-32 bg-primary/5 rounded-full blur-xl" />

      <div className="container mx-auto px-4 py-20">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div className="space-y-8">
            <div className="space-y-4">
              <div className="flex items-center space-x-2 text-primary">
                <Sparkles className="h-5 w-5" />
                <span className="text-sm font-medium">AI-Powered Shopping</span>
              </div>
              <h1 className="text-4xl md:text-6xl font-bold leading-tight">
                Smart Shopping with{" "}
                <span className="text-primary">AI Recommendations</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-lg">
                Discover products tailored just for you through AI-powered
                recommendations backed by real community data and reviews.
              </p>
            </div>

            {/* Search Bar */}
            <div className="relative max-w-md">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground h-5 w-5" />
              <Input
                disabled
                placeholder="What are you looking for today?"
                className="pl-12 pr-4 h-14 text-base border-2 border-muted focus:border-primary"
              />
              <Button
                variant="hero"
                size="lg"
                className="absolute right-2 top-2 h-10"
                onClick={onGetAIPicks}
              >
                Get AI Picks
              </Button>
            </div>

            {/* Feature Pills */}
            <div className="flex flex-wrap gap-3">
              <div className="flex items-center space-x-2 bg-card border rounded-full px-4 py-2">
                <Brain className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">AI Powered</span>
              </div>
              <div className="flex items-center space-x-2 bg-card border rounded-full px-4 py-2">
                <Users className="h-4 w-4 text-accent" />
                <span className="text-sm font-medium">Community Driven</span>
              </div>
              <div className="flex items-center space-x-2 bg-card border rounded-full px-4 py-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Personalized</span>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Button variant="hero" size="xl" className="shadow-lg">
                Start Shopping
                <Sparkles className="h-5 w-5" />
              </Button>
              {/* <Button variant="outline" size="xl"> */}
              {/*   See How It Works */}
              {/* </Button> */}
            </div>
          </div>

          {/* Right Content - Hero Image */}
          <div className="relative">
            <div className="relative z-10">
              <Player src={businessmen} loop autoplay style={{}} />
            </div>
            {/* Floating elements */}
            <div className="absolute -top-4 -right-4 bg-primary text-primary-foreground rounded-full p-3 shadow-lg animate-bounce">
              <Sparkles className="h-6 w-6" />
            </div>
            <div className="absolute -bottom-4 -left-4 bg-accent text-accent-foreground rounded-full p-3 shadow-lg animate-pulse">
              <Brain className="h-6 w-6" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
