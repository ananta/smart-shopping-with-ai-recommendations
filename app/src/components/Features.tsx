import { Card, CardContent } from "@/components/ui/card";
import {
  Brain,
  Users,
  Sparkles,
  Target,
  TrendingUp,
  Shield
} from "lucide-react";
import aiShopping from "@/assets/ai-shopping.jpg";
import { Player } from "@lottiefiles/react-lottie-player";
import shopping from "../assets/shopping.json";

const features = [
  {
    icon: Brain,
    title: "AI-Powered Recommendations",
    description:
      "Our advanced AI analyzes your preferences and shopping history to suggest products you'll love.",
    color: "text-primary",
    bgColor: "bg-primary/10"
  },
  {
    icon: Users,
    title: "Community-Driven Data",
    description:
      "Leverage real reviews and purchase patterns from millions of shoppers just like you.",
    color: "text-accent",
    bgColor: "bg-accent/10"
  },
  {
    icon: Target,
    title: "Personalized Experience",
    description:
      "Every recommendation is tailored to your unique tastes and shopping behavior.",
    color: "text-purple-500",
    bgColor: "bg-purple-500/10"
  },
  {
    icon: TrendingUp,
    title: "Trending Products",
    description:
      "Stay ahead of the curve with real-time insights on what's popular in your area.",
    color: "text-green-500",
    bgColor: "bg-green-500/10"
  },
  {
    icon: Sparkles,
    title: "Smart Search",
    description:
      "Describe what you need in natural language and let AI find the perfect products.",
    color: "text-blue-500",
    bgColor: "bg-blue-500/10"
  },
  {
    icon: Shield,
    title: "Quality Assurance",
    description:
      "Only the highest-rated products from trusted sellers make it to your recommendations.",
    color: "text-orange-500",
    bgColor: "bg-orange-500/10"
  }
];

export const Features = () => {
  return (
    <section className="py-20 bg-background">
      <div className="container mx-auto px-4">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left Content */}
          <div className="space-y-8">
            <div className="space-y-4">
              <div className="flex items-center space-x-2 text-primary">
                <Sparkles className="h-5 w-5" />
                <span className="text-sm font-medium">Why Choose DartPick</span>
              </div>
              <h2 className="text-3xl md:text-4xl font-bold">
                Shopping Powered by{" "}
                <span className="text-primary">Artificial Intelligence</span>
              </h2>
              <p className="text-lg text-muted-foreground">
                Experience the future of shopping with AI that understands your
                needs and connects you with products that matter to you.
              </p>
            </div>

            {/* Feature Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {features.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <Card
                    key={index}
                    className="border-2 hover:border-primary/20 transition-colors duration-300"
                  >
                    <CardContent className="p-6">
                      <div className="space-y-3">
                        <div
                          className={`w-12 h-12 rounded-xl ${feature.bgColor} flex items-center justify-center`}
                        >
                          <Icon className={`h-6 w-6 ${feature.color}`} />
                        </div>
                        <h3 className="font-semibold text-lg">
                          {feature.title}
                        </h3>
                        <p className="text-muted-foreground text-sm leading-relaxed">
                          {feature.description}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>

          {/* Right Content - Feature Image */}
          <div className="relative">
            <div className="relative z-10">
              <Player
                src={shopping}
                loop
                autoplay
                className="w-full max-w-lg mx-auto rounded-2xl shadow-lg"
              />
            </div>

            {/* Background decorations */}
          </div>
        </div>
      </div>
    </section>
  );
};
