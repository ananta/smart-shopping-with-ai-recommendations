import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Sparkles, Mail, Twitter, Instagram, Github } from "lucide-react";

export const Footer = () => {
  return (
    <footer className="bg-card border-t">
      <div className="container mx-auto px-4 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              {/* <div className="flex items-center justify-center w-8 h-8 bg-primary rounded-lg"> */}
              {/* <Sparkles className="h-5 w-5 text-primary-foreground" /> */}
              {/* </div> */}
              <span className="text-lg font-bold text-primary">DartPick</span>
            </div>
            <p className="text-muted-foreground text-sm leading-relaxed">
              The future of shopping is here. Let AI help you discover products
              you'll love, backed by community insights and reviews.
            </p>
            {/* <div className="flex space-x-2"> */}
            {/*   <Button variant="ghost" size="icon" className="h-8 w-8"> */}
            {/*     <Twitter className="h-4 w-4" /> */}
            {/*   </Button> */}
            {/*   <Button variant="ghost" size="icon" className="h-8 w-8"> */}
            {/*     <Instagram className="h-4 w-4" /> */}
            {/*   </Button> */}
            {/*   <Button variant="ghost" size="icon" className="h-8 w-8"> */}
            {/*     <Github className="h-4 w-4" /> */}
            {/*   </Button> */}
            {/* </div> */}
          </div>

          {/* Quick Links */}
          {/* <div className="space-y-4"> */}
          {/*   <h3 className="font-semibold">Quick Links</h3> */}
          {/*   <div className="space-y-2"> */}
          {/*     <Button */}
          {/*       variant="ghost" */}
          {/*       className="h-auto p-0 justify-start text-muted-foreground hover:text-foreground" */}
          {/*     > */}
          {/*       Browse Products */}
          {/*     </Button> */}
          {/*     <Button */}
          {/*       variant="ghost" */}
          {/*       className="h-auto p-0 justify-start text-muted-foreground hover:text-foreground" */}
          {/*     > */}
          {/*       AI Recommendations */}
          {/*     </Button> */}
          {/*     <Button */}
          {/*       variant="ghost" */}
          {/*       className="h-auto p-0 justify-start text-muted-foreground hover:text-foreground" */}
          {/*     > */}
          {/*       Community Reviews */}
          {/*     </Button> */}
          {/*     <Button */}
          {/*       variant="ghost" */}
          {/*       className="h-auto p-0 justify-start text-muted-foreground hover:text-foreground" */}
          {/*     > */}
          {/*       Trending Products */}
          {/*     </Button> */}
          {/*   </div> */}
          {/* </div> */}

          {/* Support */}
          {/* <div className="space-y-4"> */}
          {/*   <h3 className="font-semibold">Support</h3> */}
          {/*   <div className="space-y-2"> */}
          {/*     <Button */}
          {/*       variant="ghost" */}
          {/*       className="h-auto p-0 justify-start text-muted-foreground hover:text-foreground" */}
          {/*     > */}
          {/*       Help Center */}
          {/*     </Button> */}
          {/*     <Button */}
          {/*       variant="ghost" */}
          {/*       className="h-auto p-0 justify-start text-muted-foreground hover:text-foreground" */}
          {/*     > */}
          {/*       Contact Us */}
          {/*     </Button> */}
          {/*     <Button */}
          {/*       variant="ghost" */}
          {/*       className="h-auto p-0 justify-start text-muted-foreground hover:text-foreground" */}
          {/*     > */}
          {/*       Shipping Info */}
          {/*     </Button> */}
          {/*     <Button */}
          {/*       variant="ghost" */}
          {/*       className="h-auto p-0 justify-start text-muted-foreground hover:text-foreground" */}
          {/*     > */}
          {/*       Returns */}
          {/*     </Button> */}
          {/*   </div> */}
          {/* </div> */}

          {/* Newsletter */}
          {/* <div className="space-y-4"> */}
          {/*   <h3 className="font-semibold">Stay Updated</h3> */}
          {/*   <p className="text-muted-foreground text-sm"> */}
          {/*     Get AI-powered product recommendations and exclusive deals. */}
          {/*   </p> */}
          {/*   <div className="space-y-2"> */}
          {/*     <div className="flex space-x-2"> */}
          {/*       <Input placeholder="Your email" className="flex-1" /> */}
          {/*       <Button variant="cart"> */}
          {/*         <Mail className="h-4 w-4" /> */}
          {/*       </Button> */}
          {/*     </div> */}
          {/*     <p className="text-xs text-muted-foreground"> */}
          {/*       We respect your privacy. Unsubscribe at any time. */}
          {/*     </p> */}
          {/*   </div> */}
          {/* </div> */}
        </div>

        {/* Bottom Bar */}
        <div className="border-t mt-12 pt-8 flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <p className="text-muted-foreground text-sm">
            © 2025 DartPick. All rights reserved.
          </p>
          {/* <div className="flex space-x-6"> */}
          {/*   <Button */}
          {/*     variant="ghost" */}
          {/*     className="h-auto p-0 text-muted-foreground hover:text-foreground text-sm" */}
          {/*   > */}
          {/*     Privacy Policy */}
          {/*   </Button> */}
          {/*   <Button */}
          {/*     variant="ghost" */}
          {/*     className="h-auto p-0 text-muted-foreground hover:text-foreground text-sm" */}
          {/*   > */}
          {/*     Terms of Service */}
          {/*   </Button> */}
          {/*   <Button */}
          {/*     variant="ghost" */}
          {/*     className="h-auto p-0 text-muted-foreground hover:text-foreground text-sm" */}
          {/*   > */}
          {/*     Cookie Policy */}
          {/*   </Button> */}
          {/* </div> */}
        </div>
      </div>
    </footer>
  );
};
