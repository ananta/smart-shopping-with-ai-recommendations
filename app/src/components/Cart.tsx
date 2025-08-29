
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  X,
  Plus,
  Minus,
  ShoppingBag,
  Sparkles,
  Link as LinkIcon,
} from "lucide-react";
import { toast } from "sonner";

/** ---------- Legacy types (kept so existing callers don't break) ---------- */
interface CartItem {
  id: number;
  name: string;
  price: number;
  quantity: number;
  image: string;
  aiRecommended?: boolean;
}

interface CartProps {
  /** If you pass `items`, the cart renders in legacy mode; otherwise it uses the AI cart from localStorage. */
  items?: CartItem[];
  onUpdateQuantity?: (id: number, quantity: number) => void;
  onRemoveItem?: (id: number) => void;
  isOpen: boolean;
  onClose: () => void;
  /** Optional: show budget usage if you want to display a bar up top */
  budget?: number;
}

/** ---------- AI cart (from ProductGrid) ---------- */
const CART_STORAGE_KEY = "ai_cart_v1";

type Retailer = { name?: string; url?: string; price?: number; currency?: string };
type AICartItem = {
  __id: string;
  __category: string;
  product: string;
  best_price?: number | null;
  currency?: string | null;
  agent_decision?: "buy" | "wait" | "consider alt" | null;
  votes?: number;
  why?: string;
  highlights?: string[];
  attributes?: Record<string, any>;
  retailers?: Retailer[];
};

type AICartMap = Record<string, AICartItem>; // key = category

const fmtMoney = (amount?: number | null, currency?: string | null) => {
  if (amount == null || Number.isNaN(Number(amount))) return "—";
  const cur = currency || "USD";
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency: cur }).format(Number(amount));
  } catch {
    return `${cur} ${Number(amount).toFixed(2)}`;
  }
};

export const Cart = ({
  items,
  onUpdateQuantity,
  onRemoveItem,
  isOpen,
  onClose,
  budget,
}: CartProps) => {
  if (!isOpen) return null;

  /** ---------- AI cart state (only used if `items` not provided) ---------- */
  const [aiCart, setAiCart] = useState<AICartMap>({});

  // Load AI cart on mount/open
  useEffect(() => {
    if (items && items.length) return; // legacy mode; ignore aiCart
    try {
      const raw = localStorage.getItem(CART_STORAGE_KEY);
      if (raw) setAiCart(JSON.parse(raw));
      else setAiCart({});
    } catch {
      setAiCart({});
    }
  }, [items, isOpen]);

  // Sync across tabs/windows
  useEffect(() => {
    if (items && items.length) return;
    const onStorage = (e: StorageEvent) => {
      if (e.key === CART_STORAGE_KEY) {
        try {
          setAiCart(e.newValue ? JSON.parse(e.newValue) : {});
        } catch {
          setAiCart({});
        }
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [items]);

  const writeAiCart = (next: AICartMap) => {
    setAiCart(next);
    try {
      localStorage.setItem(CART_STORAGE_KEY, JSON.stringify(next));
    } catch {}
  };

  const removeAICategory = (category: string) => {
    const { [category]: _, ...rest } = aiCart;
    writeAiCart(rest);
  };

  const clearAICart = () => writeAiCart({});

  const copyAISummary = async () => {
    const entries = Object.entries(aiCart).sort(([a], [b]) => a.localeCompare(b));
    const lines: string[] = [];
    let total = 0;
    for (const [cat, it] of entries) {
      const p = typeof it.best_price === "number" ? it.best_price : 0;
      total += p;
      lines.push(`• ${cat}: ${it.product} — ${fmtMoney(p, it.currency ?? "USD")}`);
    }
    lines.push(`\nTotal: ${fmtMoney(total, "USD")}${budget ? ` / Budget: ${fmtMoney(budget, "USD")}` : ""}`);
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      toast.success("Cart copied to clipboard");
    } catch {}
  };

  /** ---------- Totals ---------- */
  const legacySubtotal = useMemo(
    () => (items?.length ? items.reduce((s, it) => s + it.price * it.quantity, 0) : 0),
    [items]
  );

  const aiSubtotal = useMemo(() => {
    const arr = Object.values(aiCart);
    return arr.reduce((s, it) => s + (typeof it.best_price === "number" ? it.best_price : 0), 0);
  }, [aiCart]);

  const subtotal = items?.length ? legacySubtotal : aiSubtotal;
  const shipping = subtotal > 50 ? 0 : 5.99;
  const total = subtotal + (subtotal > 0 ? shipping : 0);

  const pctBudget = budget ? Math.min(100, Math.round((subtotal / budget) * 100)) : null;

  /** ---------- UI ---------- */
  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm">
      <div className="fixed right-0 top-0 h-full w-full max-w-md bg-background shadow-2xl">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b">
            <div className="flex items-center space-x-2">
              <ShoppingBag className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Your Cart</h2>
              <Badge variant="secondary">
                {items?.length ?? Object.keys(aiCart).length}
              </Badge>
            </div>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-5 w-5" />
            </Button>
          </div>

          {/* Optional budget usage bar (AI mode) */}
          {budget && !items?.length && (
            <div className="px-6 pt-4">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Budget usage</span>
                <span className="font-medium">
                  {fmtMoney(subtotal, "USD")} {pctBudget != null ? `(${pctBudget}%)` : ""}
                </span>
              </div>
              <div className="w-full h-2 bg-muted rounded-full overflow-hidden mt-1">
                <div
                  className={`h-full ${pctBudget && pctBudget > 100 ? "bg-red-500" : "bg-primary"}`}
                  style={{ width: `${Math.min(100, pctBudget ?? 0)}%` }}
                />
              </div>
            </div>
          )}

          {/* Body */}
          <div className="flex-1 overflow-y-auto p-6">
            {/* Legacy cart mode */}
            {items?.length ? (
              <div className="space-y-4">
                {items.map((item) => (
                  <Card key={item.id} className="overflow-hidden">
                    <CardContent className="p-4">
                      <div className="flex space-x-3">
                        <img
                          src={item.image}
                          alt={item.name}
                          className="w-16 h-16 object-cover rounded-lg"
                        />
                        <div className="flex-1 space-y-2">
                          <div className="flex items-start justify-between">
                            <div>
                              <h4 className="font-medium text-sm leading-tight line-clamp-2">
                                {item.name}
                              </h4>
                              {item.aiRecommended && (
                                <Badge className="mt-1 bg-primary/10 text-primary border-primary/20">
                                  <Sparkles className="h-3 w-3 mr-1" />
                                  AI Pick
                                </Badge>
                              )}
                            </div>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 text-muted-foreground hover:text-destructive"
                              onClick={() => onRemoveItem?.(item.id)}
                            >
                              <X className="h-4 w-4" />
                            </Button>
                          </div>

                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-primary">
                              ${(item.price * item.quantity).toFixed(2)}
                            </span>

                            <div className="flex items-center space-x-2">
                              <Button
                                variant="outline"
                                size="icon"
                                className="h-7 w-7"
                                onClick={() => onUpdateQuantity?.(item.id, Math.max(1, item.quantity - 1))}
                              >
                                <Minus className="h-3 w-3" />
                              </Button>
                              <span className="w-8 text-center text-sm font-medium">
                                {item.quantity}
                              </span>
                              <Button
                                variant="outline"
                                size="icon"
                                className="h-7 w-7"
                                onClick={() => onUpdateQuantity?.(item.id, item.quantity + 1)}
                              >
                                <Plus className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              // AI cart mode: one selection per category (from ProductGrid)
              <>
                {Object.keys(aiCart).length === 0 ? (
                  <div className="text-center py-12">
                    <ShoppingBag className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-medium mb-2">Your cart is empty</h3>
                    <p className="text-muted-foreground">
                      Pick one item per category from the AI recommendations.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {Object.entries(aiCart)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([category, item]) => (
                        <Card key={item.__id} className="overflow-hidden">
                          <CardContent className="p-4">
                            <div className="flex gap-3">
                              {/* Placeholder visual (no images in AI picks yet) */}
                              <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-muted to-background flex items-center justify-center">
                                <span className="text-xs text-muted-foreground text-center">
                                  {category}
                                </span>
                              </div>

                              <div className="flex-1 space-y-2">
                                <div className="flex items-start justify-between">
                                  <div>
                                    <div className="text-xs uppercase text-muted-foreground">
                                      {category}
                                    </div>
                                    <h4 className="font-medium text-sm leading-tight line-clamp-2">
                                      {item.product}
                                    </h4>
                                    {item.agent_decision && (
                                      <Badge variant="outline" className="mt-1 capitalize">
                                        {item.agent_decision}
                                      </Badge>
                                    )}
                                  </div>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-6 w-6 text-muted-foreground hover:text-destructive"
                                    onClick={() => removeAICategory(category)}
                                  >
                                    <X className="h-4 w-4" />
                                  </Button>
                                </div>

                                {/* Retailers (up to 2) */}
                                {Array.isArray(item.retailers) && item.retailers.length > 0 && (
                                  <div className="flex flex-wrap gap-2">
                                    {item.retailers.slice(0, 2).map((r, idx) => (
                                      <a
                                        key={idx}
                                        href={r.url || "#"}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md border hover:bg-accent"
                                      >
                                        <LinkIcon className="w-3 h-3" />
                                        <span>{r.name || "Retailer"}</span>
                                        {typeof r.price === "number" && (
                                          <span className="font-medium">
                                            · {fmtMoney(r.price, r.currency ?? item.currency ?? undefined)}
                                          </span>
                                        )}
                                      </a>
                                    ))}
                                  </div>
                                )}

                                <div className="flex items-center justify-between">
                                  <span className="font-semibold text-primary">
                                    {fmtMoney(item.best_price ?? null, item.currency ?? undefined)}
                                  </span>
                                  {/* No quantity controls in AI mode (1 per category) */}
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                  </div>
                )}
              </>
            )}
          </div>

          {/* Footer */}
          {subtotal > 0 && (
            <div className="border-t p-6 space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Subtotal</span>
                  <span>{fmtMoney(subtotal, "USD")}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Shipping</span>
                  <span>{shipping === 0 ? "Free" : fmtMoney(shipping, "USD")}</span>
                </div>
                {shipping > 0 && (
                  <p className="text-xs text-muted-foreground">
                    Free shipping on orders over $50
                  </p>
                )}
                <Separator />
                <div className="flex justify-between font-semibold text-lg">
                  <span>Total</span>
                  <span className="text-primary">{fmtMoney(total, "USD")}</span>
                </div>
              </div>

              {/* AI-only actions */}
              {!items?.length && (
                <div className="flex gap-2">
                  <Button variant="outline" className="flex-1" onClick={copyAISummary}>
                    Copy summary
                  </Button>
                  <Button variant="ghost" className="flex-1" onClick={clearAICart}>
                    Clear cart
                  </Button>
                </div>
              )}

              <Button
                variant="hero"
                className="w-full shadow-lg"
                onClick={() =>
                  toast.success("Redirecting to checkout...", {
                    description: "Your AI-curated cart is ready for purchase!",
                  })
                }
              >
                <Sparkles className="h-4 w-4 mr-2" />
                Checkout with AI
              </Button>

              <Button variant="outline" className="w-full" onClick={onClose}>
                Continue Shopping
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};



