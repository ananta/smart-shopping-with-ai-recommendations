import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sparkles, Link as LinkIcon } from "lucide-react";
import { toast } from "sonner";


const AI_CART_KEY = "ai_cart_v1";

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
  retailers?: { name?: string; url?: string; price?: number; currency?: string }[];
};

const loadAiCart = (): Record<string, AICartItem> => {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(AI_CART_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
};

const saveAiCart = (map: Record<string, AICartItem>) => {
  localStorage.setItem(AI_CART_KEY, JSON.stringify(map));
  // notify same-tab listeners (Cart already listens to storage across tabs)
  window.dispatchEvent(new CustomEvent("ai_cart:changed"));
};

const vendorFromUrl = (url?: string | null) => {
  try {
    if (!url) return;
    const host = new URL(url).hostname.replace(/^www\./, "");
    const name = host.split(".")[0];
    return name ? name.charAt(0).toUpperCase() + name.slice(1) : undefined;
  } catch { /* noop */ }
};

const offersToRetailers = (c: any) => {
  const best = c.best_offer
    ? [{
      name: c.best_offer.vendor ?? vendorFromUrl(c.best_offer.affiliate_url || c.best_offer.url) ?? "Vendor",
      url: c.best_offer.affiliate_url || c.best_offer.url || undefined,
      price: typeof c.best_offer.price === "number" ? c.best_offer.price : undefined,
      currency: c.best_offer.currency ?? c.currency ?? "USD",
    }]
    : [];

  const others = Array.isArray(c.offers) ? c.offers.map((o: any) => ({
    name: o?.vendor ?? vendorFromUrl(o?.affiliate_url || o?.url) ?? "Retailer",
    url: o?.affiliate_url || o?.url || undefined,
    price: typeof o?.price === "number" ? o.price : undefined,
    currency: o?.currency ?? c.currency ?? "USD",
  })) : [];

  // dedupe by name+url
  const seen = new Set<string>();
  const all = [...best, ...others].filter((r) => {
    const k = `${r.name}|${r.url}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });

  return all;
};

const upsertAiCart = (category: string, entry: AICartItem) => {
  const map = loadAiCart();
  // one item per category (replace)
  map[category] = entry;
  saveAiCart(map);
};


type Retailer = { name?: string; url?: string; price?: number; currency?: string };
type TopCommentItem = {
  product: string;
  votes?: number;
  best_price?: number | null;
  currency?: string | null;
  agent_decision?: "buy" | "wait" | "consider alt" | null;
  why?: string;
  highlights?: string[];
  attributes?: Record<string, any>;
  retailers?: Retailer[];
};

type CategoryRec = { top_comments?: TopCommentItem[] };
type Recs = Record<string, CategoryRec>; // { "Espresso Machines": { top_comments: [...] }, ... }

const CART_STORAGE_KEY = "ai_cart_v1";

// ===== Utilities =====
const fmtMoney = (amount?: number | null, currency?: string | null) => {
  if (amount == null || Number.isNaN(Number(amount))) return "—";
  const cur = currency || "USD";
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency: cur }).format(Number(amount));
  } catch {
    return `${cur} ${Number(amount).toFixed(2)}`;
  }
};

const slugId = (category: string, product: string) =>
  `${category}::${product}`.toLowerCase().replace(/\s+/g, "-");

// Cosmetic star number if your backend doesn’t provide one
const starFromDecision = (d?: string | null) =>
  d === "buy" ? 4.8 : d === "consider alt" ? 4.3 : d === "wait" ? 3.9 : 4.0;

// Deduplicate within a category by product
const dedupeByProduct = (items: TopCommentItem[]) => {
  const map = new Map<string, TopCommentItem>();
  for (const it of items) {
    const key = (it.product || "").trim().toLowerCase();
    if (!map.has(key)) map.set(key, it);
  }
  return Array.from(map.values());
};

// Flatten recs → grid items (keep source category for cart)
type GridItem = TopCommentItem & { __id: string; __category: string };
const flattenRecs = (recs: Recs): GridItem[] => {
  const out: GridItem[] = [];
  Object.entries(recs || {}).forEach(([category, rec]) => {
    const items = dedupeByProduct(rec?.top_comments || []);
    // Sort: decision → price → votes (same as the earlier recs panel)
    items.sort((a, b) => {
      const order = (d?: string | null) => (d === "buy" ? 0 : d === "consider alt" ? 1 : d === "wait" ? 2 : 3);
      const byDecision = order(a.agent_decision) - order(b.agent_decision);
      if (byDecision !== 0) return byDecision;
      const pa = a.best_price ?? Number.POSITIVE_INFINITY;
      const pb = b.best_price ?? Number.POSITIVE_INFINITY;
      if (pa !== pb) return pa - pb;
      return (b.votes ?? 0) - (a.votes ?? 0);
    });
    items.forEach((it) => out.push({ ...it, __id: slugId(category, it.product), __category: category }));
  });
  return out;
};

interface ProductGridProps {
  recommendations: Recs;
  onAddToCartOptional?: (item: GridItem) => void; // optional analytics hook
}


export function ProductGrid({
  recommendations,
  onAddToCartOptional,
}: {
    recommendations: Record<string, { top_comments?: any[] }>;
    onAddToCartOptional?: (item: any) => void;
  }) {
  const fmtMoney = (amount: number | null | undefined, currency?: string | null) => {
    if (amount == null || Number.isNaN(Number(amount))) return "—";
    const cur = currency || "USD";
    try {
      return new Intl.NumberFormat(undefined, { style: "currency", currency: cur }).format(Number(amount));
    } catch {
      return `${cur} ${Number(amount).toFixed(2)}`;
    }
  };

  const cats = Object.keys(recommendations || {});
  if (!cats.length) return <div className="text-sm text-muted-foreground">Waiting for retrieval…</div>;

  return (
    <div className="space-y-4">
      {cats.map((cat) => {
        const items = recommendations[cat]?.top_comments ?? [];
        return (
          <div key={cat} className="border rounded-xl p-4 bg-card">
            <div className="font-semibold text-card-foreground mb-3">{cat}</div>
            {!items.length ? (
              <div className="text-sm text-muted-foreground">No items yet…</div>
            ) : (
                <div className="grid md:grid-cols-2 gap-3">
                  {items.map((c: any, idx: number) => (
                    <div key={`${c.product}-${idx}`} className="flex flex-col gap-3 border rounded-lg p-3 bg-background/50">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1">
                          <div className="font-medium text-foreground">{c.product}</div>
                          {typeof c.votes === "number" && (
                            <div className="text-xs text-muted-foreground mt-1">
                              <Sparkles className="w-3 h-3 inline mr-1" />
                              {c.votes} community votes
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="text-lg font-bold text-primary">
                            {fmtMoney(c.best_offer?.price ?? c.best_price ?? null, c.best_offer?.currency ?? c.currency)}
                          </div>
                          {c.agent_decision && (
                            <Badge
                              variant={
                                c.agent_decision === "buy"
                                  ? "default"
                                  : c.agent_decision === "wait"
                                    ? "secondary"
                                    : "outline"
                              }
                            >
                              {c.agent_decision}
                            </Badge>
                          )}
                        </div>
                      </div>

                      {(c.best_offer || (Array.isArray(c.offers) && c.offers.length > 0)) && (
                        <div className="space-y-2">
                          {c.best_offer && (
                            <div className="flex flex-wrap items-center gap-2 text-sm">
                              <Badge className="uppercase">Best price</Badge>
                              <a
                                href={(c.best_offer.url || "#") as string}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-1 underline-offset-2 hover:underline"
                              >
                                <LinkIcon className="w-3 h-3" />
                                <span>{c.best_offer.vendor ?? "Vendor"}</span>
                              </a>
                              <span className="font-semibold">
                                {fmtMoney(c.best_offer.price ?? null, c.best_offer.currency ?? c.currency ?? "USD")}
                              </span>
                              {c.best_offer.coupon && <Badge variant="secondary">Coupon: {c.best_offer.coupon}</Badge>}
                              {c.best_offer.risk && <Badge variant="destructive">{c.best_offer.risk}</Badge>}
                            </div>
                          )}

                          {Array.isArray(c.offers) && c.offers.length > 0 && (
                            <div className="flex flex-wrap gap-2">
                              {c.offers.map((o: any, i: number) => (
                                <a
                                  key={i}
                                  href={(o.url || "#") as string}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md border hover:bg-accent"
                                >
                                  <LinkIcon className="w-3 h-3" />
                                  <span>{o.vendor ?? "Retailer"}</span>
                                  {typeof o.price === "number" && (
                                    <span className="font-medium">· {fmtMoney(o.price, o.currency ?? c.currency ?? "USD")}</span>
                                  )}
                                  {o.coupon && <span className="ml-1 opacity-80">(coupon: {o.coupon})</span>}
                                </a>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          className="px-3"
                          onClick={() => {
                            const price = typeof c.best_offer?.price === "number" ? c.best_offer.price
                              : typeof c.best_price === "number" ? c.best_price
                                : null;

                            const aiItem: AICartItem = {
                              __id: `${cat}::${c.product}`.toLowerCase().replace(/\s+/g, "-"),
                              __category: cat,
                              product: c.product,
                              best_price: price,
                              currency: c.best_offer?.currency ?? c.currency ?? "USD",
                              agent_decision: c.agent_decision ?? null,
                              votes: c.votes,
                              why: c.why,
                              highlights: Array.isArray(c.highlights) ? c.highlights : undefined,
                              attributes: c.attributes ?? undefined,
                              retailers: offersToRetailers(c), // includes best + others
                            };
                            upsertAiCart(cat, aiItem);
                            onAddToCartOptional?.(aiItem);
                            try { (toast as any)?.success?.("Added to cart"); } catch {}
                          }}
                        >
                          Add to cart
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
          </div>
        );
      })}
    </div>
  );
}



