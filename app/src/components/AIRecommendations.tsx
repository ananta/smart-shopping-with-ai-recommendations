import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { X, Sparkles, Clock, CheckCircle, Loader2, ChevronDown, ChevronUp, Link as LinkIcon, Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ProductGrid } from "./ProductGrid";

// If your API is not on the same origin, set this to e.g. "http://127.0.0.1:8000"
const API_BASE = "http://localhost:8000"; // same-origin proxy; else set explicit origin

// Known graph nodes in expected order (for a nice progress timeline)
const KNOWN_STEPS = [
  { id: "goal_analyzer", label: "Analyze goal" },
  { id: "ai_spec_builder", label: "Build spec" },
  { id: "retriever", label: "Retrieve candidates" },
  { id: "pricing_enricher", label: "Get pricing" },
  { id: "ai_attribute_extractor", label: "Extract attributes" },
  { id: "ai_decision_maker", label: "Make decision" },
  { id: "synthesizer", label: "Write summary" },
];

type Retailer = { name?: string; url?: string; price?: number; currency?: string };


type PublicSlice = {
  category_plan?: string[];
  spec?: Record<string, any>;
  recommendations?: Record<string, CategoryRec>;
  final_output?: string | null;
};

type NodeEndEvent = {
  node: string;
  state: PublicSlice;
  ts: number; // epoch seconds (float)
};

type StartEvent = { user_goal: string; ts: number };
type DoneEvent = { state: PublicSlice; ts: number };

interface AIRecommendationsProps {
  isOpen: boolean;
  onClose: () => void;
  initialGoal?: string;
}


type Offer = {
  vendor?: string | null;
  price?: number | null;
  currency?: string | null;
  url?: string | null;
  affiliate_url?: string | null;
  coupon?: string | null;
  risk?: string | null; // e.g. "3p-seller", "refurb", etc.
};

type TopCommentItem = {
  product: string;
  votes?: number;
  best_price?: number | null;
  currency?: string | null;
  agent_decision?: "buy" | "wait" | "consider alt" | null;
  why?: string;
  highlights?: string[];
  attributes?: Record<string, any>;
  // NEW:
  best_offer?: Offer | null;
  offers?: Offer[];
};

type CategoryRec = { top_comments?: TopCommentItem[] }; // (unchanged)

// If you want to keep Retailer type for other code paths, leave it;
// but we’ll render from Offer/best_offer going forward.

const vendorFromUrl = (url?: string | null) => {
  try {
    if (!url) return undefined;
    const host = new URL(url).hostname.replace(/^www\./, "");
    const first = host.split(".")[0];
    return first ? first.charAt(0).toUpperCase() + first.slice(1) : undefined;
  } catch {
    return undefined;
  }
};

// Normalize server fields into a stable UI shape
const normalizeRecs = (incoming?: Record<string, CategoryRec>): Record<string, CategoryRec> => {
  const out: Record<string, CategoryRec> = {};
  if (!incoming) return out;

  for (const [cat, rec] of Object.entries(incoming)) {
    const items = Array.isArray(rec?.top_comments) ? rec.top_comments : [];
    out[cat] = {
      top_comments: items.map((it: any) => {
        const offers: Offer[] = (Array.isArray(it.offers) ? it.offers : []).map((o: any) => ({
          vendor: o?.vendor ?? vendorFromUrl(o?.affiliate_url || o?.url) ?? "Unknown",
          price: typeof o?.price === "number" ? o.price : null,
          currency: o?.currency ?? it?.currency ?? "USD",
          url: o?.affiliate_url || o?.url || null,
          affiliate_url: o?.affiliate_url ?? null,
          coupon: o?.coupon ?? null,
          risk: o?.risk ?? null,
        }));

        const best: Offer | null = it?.best_offer
          ? {
              vendor: it.best_offer.vendor ?? vendorFromUrl(it.best_offer.affiliate_url || it.best_offer.url) ?? "Unknown",
              price: typeof it.best_offer.price === "number" ? it.best_offer.price : null,
              currency: it.best_offer.currency ?? it.currency ?? "USD",
              url: it.best_offer.affiliate_url || it.best_offer.url || null,
              affiliate_url: it.best_offer.affiliate_url ?? null,
              coupon: it.best_offer.coupon ?? null,
              risk: it.best_offer.risk ?? null,
            }
          : null;

        return {
          ...it,
          // ensure best_price/currency are always present for UI math
          best_price: typeof it.best_price === "number" ? it.best_price : best?.price ?? null,
          currency: it?.currency ?? best?.currency ?? "USD",
          best_offer: best,
          offers,
        } as TopCommentItem;
      }),
    };
  }
  return out;
};


export const AIRecommendations = ({ isOpen, onClose, initialGoal = "" }: AIRecommendationsProps) => {
  const [goal, setGoal] = useState(initialGoal);
  const [isStreaming, setIsStreaming] = useState(false);
  const [categories, setCategories] = useState<string[]>([]);
  const [spec, setSpec] = useState<Record<string, any>>({});
  const [recs, setRecs] = useState<Record<string, CategoryRec>>({});
  const [summary, setSummary] = useState<string>("");
  const [errors, setErrors] = useState<string | null>(null);

  // Progress state
  const [timeline, setTimeline] = useState<Record<string, "pending" | "done" | "active">>({});
  const [nodeTimes, setNodeTimes] = useState<Record<string, number>>({}); // ts per node
  const [startTs, setStartTs] = useState<number | null>(null);
  const [endTs, setEndTs] = useState<number | null>(null);

  // Debug raw view
  const [showRaw, setShowRaw] = useState(false);
  const [lastState, setLastState] = useState<PublicSlice>({});

  const esRef = useRef<EventSource | null>(null);
  const summaryRef = useRef<HTMLDivElement>(null);

  // Initialize timeline based on known steps
  const initialTimeline = useMemo(() => {
    const map: Record<string, "pending" | "done" | "active"> = {};
    KNOWN_STEPS.forEach((s, idx) => (map[s.id] = idx === 0 ? "active" : "pending"));
    return map;
  }, []);

  useEffect(() => {
    // auto-scroll summary into view when updated
    if (summary && summaryRef.current) {
      summaryRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [summary]);

  useEffect(() => {
    setGoal(initialGoal);
  }, [initialGoal]);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  }, []);

  const resetUI = () => {
    setCategories([]);
    setSpec({});
    setRecs({});
    setSummary("");
    setErrors(null);
    setTimeline(initialTimeline);
    setNodeTimes({});
    setStartTs(null);
    setEndTs(null);
    setLastState({});
  };

  const fmtMoney = (amount: number | null | undefined, currency?: string | null) => {
    if (amount == null || Number.isNaN(Number(amount))) return "—";
    const cur = currency || "USD";
    try {
      return new Intl.NumberFormat(undefined, { style: "currency", currency: cur }).format(Number(amount));
    } catch {
      return `${cur} ${Number(amount).toFixed(2)}`;
    }
  };

  const computeBudgetUsage = (recs: Record<string, CategoryRec>, budget?: number) => {
    const chosenPerCategory = Object.values(recs).map((cat) => {
      const items = (cat.top_comments || []).filter(Boolean);
      if (!items.length) return 0;
      // prefer items the agent said "buy", otherwise min priced
      const buyItems = items.filter(i => i.agent_decision === "buy" && i.best_price != null);
      const selected = (buyItems.length ? buyItems : items).reduce((min, it) => {
        const price = (it.best_price ?? Number.POSITIVE_INFINITY) as number;
        return price < (min?.best_price ?? Number.POSITIVE_INFINITY) ? it : min;
      }, undefined as TopCommentItem | undefined);
      return selected?.best_price ? Number(selected.best_price) : 0;
    });
    const total = chosenPerCategory.reduce((a, b) => a + b, 0);
    const pct = budget ? Math.min(100, Math.round((total / budget) * 100)) : null;
    return { total, pct };
  };

  const start = () => {
    if (!goal.trim()) {
      setErrors("Please enter a goal.");
      return;
    }
    resetUI();

    const url = `${API_BASE}/recommend/stream?user_goal=${encodeURIComponent(goal)}`;
    const es = new EventSource(url);
    esRef.current = es;
    setIsStreaming(true);

    es.addEventListener("start", (evt) => {
      const data: StartEvent = JSON.parse((evt as MessageEvent).data);
      setStartTs(data.ts);
    });

    es.addEventListener("node_end", (evt) => {
      const { node, state, ts } = JSON.parse((evt as MessageEvent).data) as NodeEndEvent;

      // Mark timeline
      setTimeline((prev) => {
        const next = { ...prev };
        if (node in next) next[node] = "done";
        const idx = KNOWN_STEPS.findIndex((s) => s.id === node);
        if (idx >= 0 && idx + 1 < KNOWN_STEPS.length) {
          const nextId = KNOWN_STEPS[idx + 1].id;
          if (next[nextId] === "pending") next[nextId] = "active";
        }
        return next;
      });


      // Capture timestamps
      setNodeTimes((prev) => ({ ...prev, [node]: ts }));

        if (state.category_plan) setCategories(state.category_plan);
        if (state.spec) setSpec(state.spec);
        if (state.recommendations) setRecs(normalizeRecs(state.recommendations)); // <-- important
        if (state.final_output) setSummary(state.final_output || "");
        setLastState(state);
    });

    es.addEventListener("done", (evt) => {
      const data = JSON.parse((evt as MessageEvent).data) as DoneEvent;
      if (data.state?.final_output) setSummary(data.state.final_output);
      if (data.state?.recommendations) setRecs(normalizeRecs(data.state.recommendations)); // <-- normalize here too
      setLastState(data.state || {});


      setEndTs(data.ts ?? Date.now() / 1000);

      // Ensure all steps show as done
      setTimeline((prev) => {
        const next = { ...prev };
        KNOWN_STEPS.forEach((s) => (next[s.id] = "done"));
        return next;
      });

      setIsStreaming(false);
      es.close();
      esRef.current = null;
    });

    es.onerror = (e) => {
      console.error("SSE error", e);
      setErrors("Stream connection lost or server error.");
      setIsStreaming(false);
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  };

  const stop = () => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setIsStreaming(false);
  };

  // Render helpers
  const StepBadge: React.FC<{ status: "pending" | "done" | "active" }> = ({ status }) => {
    if (status === "done") return <CheckCircle className="w-4 h-4 text-green-600" />;
    if (status === "active") return <Loader2 className="w-4 h-4 text-primary animate-spin" />;
    return <Clock className="w-4 h-4 text-muted-foreground" />;
  };

  const stepDuration = (idx: number) => {
    const cur = KNOWN_STEPS[idx]?.id;
    const prev = KNOWN_STEPS[idx - 1]?.id;
    const curTs = cur ? nodeTimes[cur] : undefined;
    const prevTs = idx === 0 ? startTs : prev ? nodeTimes[prev] : undefined;
    if (!curTs || !prevTs) return null;
    const secs = Math.max(0, curTs - prevTs);
    return `${secs.toFixed(1)}s`;
    // Note: backend ts looks like epoch seconds w/ fraction in your sample
  };

  const overallElapsed = () => {
    if (!startTs) return null;
    const end = endTs ?? Date.now() / 1000;
    const secs = Math.max(0, end - startTs);
    return `${secs.toFixed(1)}s`;
  };

  const dedupeAndSort = (items: TopCommentItem[]) => {
    // Deduplicate by product (case-insensitive)
    const map = new Map<string, TopCommentItem>();
    for (const it of items) {
      const key = (it.product || "").trim().toLowerCase();
      if (!map.has(key)) map.set(key, it);
    }
    const arr = Array.from(map.values());
    // Decision -> price -> votes
    return arr.sort((a, b) => {
      const order = (d?: string | null) => (d === "buy" ? 0 : d === "consider alt" ? 1 : d === "wait" ? 2 : 3);
      const byDecision = order(a.agent_decision) - order(b.agent_decision);
      if (byDecision !== 0) return byDecision;
      const pa = a.best_price ?? Number.POSITIVE_INFINITY;
      const pb = b.best_price ?? Number.POSITIVE_INFINITY;
      if (pa !== pb) return pa - pb;
      return (b.votes ?? 0) - (a.votes ?? 0);
    });
  };

  const renderRecs = () => {
    const keys = Object.keys(recs || {});
    if (!keys.length) return <div className="text-sm text-muted-foreground">Waiting for retrieval…</div>;
    return (
      <div className="space-y-4">
        {keys.map((cat) => {
          const val = recs[cat];
          const itemsRaw = Array.isArray(val?.top_comments) ? val!.top_comments! : [];
          const items = dedupeAndSort(itemsRaw as TopCommentItem[]);
          return (
            <div key={cat} className="border rounded-xl p-4 bg-card">
              <div className="font-semibold text-card-foreground mb-3">{cat}</div>
              {!items.length ? (
                <div className="text-sm text-muted-foreground">No items yet…</div>
              ) : (
                <div className="space-y-3">
                  {items.map((c, idx) => (
                    <div
                      key={`${c.product}-${idx}`}
                      className="flex flex-col gap-3 border rounded-lg p-3 bg-background/50"
                    >
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                        <div className="flex-1">
                          <div className="font-medium text-foreground">{c.product}</div>
                          {typeof c.votes === "number" && (
                            <div className="text-xs text-muted-foreground mt-1">
                              <Sparkles className="w-3 h-3 inline mr-1" />
                              {c.votes} community votes
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="text-lg font-bold text-primary">
                            {fmtMoney(c.best_price ?? null, c.currency ?? undefined)}
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

                      {/* Why & highlights if provided by backend */}
                      {(c.why || (c.highlights && c.highlights.length)) && (
                        <div className="text-sm text-foreground/90">
                          {c.why && <div className="mb-1"><span className="font-medium">Why:</span> {c.why}</div>}
                          {c.highlights && c.highlights.length > 0 && (
                            <ul className="list-disc pl-5 space-y-1">
                              {c.highlights.map((h, i) => (
                                <li key={i}>{h}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )}

                      {/* Attributes if provided */}
                      {c.attributes && Object.keys(c.attributes).length > 0 && (
                        <div className="text-xs text-muted-foreground grid sm:grid-cols-2 gap-x-6 gap-y-1">
                          {Object.entries(c.attributes).map(([k, v]) => (                            <div key={k} className="flex justify-between gap-3">
                              <span className="capitalize">{k.replace(/_/g, " ")}</span>
                              <span className="text-foreground">{String(v)}</span>
                            </div>
                          ))}
                        </div>
                      )}


                        {/* Best vendor + other offers */}
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
        {c.offers.map((o, i) => (
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
                        <Button size="sm" className="px-3">Add to cart</Button>
                        <Button size="sm" variant="secondary" className="px-3">Save</Button>
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
  };

  if (!isOpen) return null;

  const budget = typeof spec?.budget === "number" ? spec.budget : undefined;
  const { total: totalChosen, pct } = computeBudgetUsage(recs, budget);

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="bg-background rounded-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden shadow-2xl">
        <div className="flex items-center justify-between p-6 border-b bg-card">
          <div className="flex items-center gap-3">
            <Sparkles className="w-6 h-6 text-primary" />
            <h2 className="text-2xl font-bold text-card-foreground">AI Shopping Assistant</h2>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-5 h-5" />
          </Button>
        </div>

        <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(90vh-80px)]">
          {/* Goal entry */}
          <div className="flex flex-col sm:flex-row gap-3">
            <Input
              className="flex-1"
              placeholder="e.g. I want a budget home espresso setup under $500"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              disabled={isStreaming}
            />
            <div className="flex gap-2">
              {!isStreaming ? (
                <Button onClick={start} className="px-6">
                  <Sparkles className="w-4 h-4 mr-2" />
                  Get AI Recommendations
                </Button>
              ) : (
                <Button variant="secondary" onClick={stop} className="px-6">
                  Stop Analysis
                </Button>
              )}
            </div>
          </div>

          {errors && (
            <div className="rounded-xl bg-destructive/10 text-destructive p-4 text-sm border border-destructive/20">
              {errors}
            </div>
          )}

          {/* Progress timeline with per-step durations */}
          <div className="border rounded-xl p-4 bg-card">
            <div className="flex items-center justify-between mb-4">
              <div className="font-semibold text-card-foreground">Analysis Progress</div>
              <div className="text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1">
                  <Info className="w-3 h-3" /> Elapsed: {overallElapsed() ?? "—"}
                </span>
              </div>
            </div>
            <div className="space-y-3">
              {KNOWN_STEPS.map((s, idx) => (
                <div key={s.id} className="flex items-center gap-3">
                  <StepBadge status={timeline[s.id] || "pending"} />
                  <span
                    className={`text-sm ${
                      timeline[s.id] === "done"
                        ? "text-foreground"
                        : timeline[s.id] === "active"
                        ? "text-primary"
                        : "text-muted-foreground"
                    }`}
                  >
                    {s.label}
                  </span>
                  <span className="text-xs text-muted-foreground ml-auto">
                    {stepDuration(idx) ?? ""}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Categories & Spec */}
          <div className="grid sm:grid-cols-3 gap-4">
            <div className="border rounded-xl p-4 bg-card sm:col-span-1">
              <div className="font-semibold text-card-foreground mb-3">Categories</div>
              {!categories.length ? (
                <div className="text-sm text-muted-foreground">Waiting for goal analysis…</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {categories.map((c) => (
                    <Badge key={c} variant="secondary">
                      {c}
                    </Badge>
                  ))}
                </div>
              )}
            </div>

            <div className="border rounded-xl p-4 bg-card sm:col-span-2 space-y-3">
              <div className="font-semibold text-card-foreground">Specification</div>
              {!Object.keys(spec).length ? (
                <div className="text-sm text-muted-foreground">Waiting for spec builder…</div>
              ) : (
                <>
                  <div className="grid sm:grid-cols-2 gap-x-6 gap-y-2 text-sm">
                    {Object.entries(spec).map(([key, value]) => (
                      <div key={key} className="flex justify-between items-center">
                        <span className="text-muted-foreground capitalize">{key.replace(/_/g, " ")}</span>
                        <span className="font-medium text-foreground">
                          {Array.isArray(value)
                            ? value.join(", ")
                            : typeof value === "boolean"
                            ? value ? "Yes" : "No"
                            : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                  {typeof spec?.budget === "number" && (
                    <div className="mt-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Budget usage</span>
                        <span className="font-medium">
                          {fmtMoney(totalChosen, "USD")} {pct != null ? `(${pct}%)` : ""}
                        </span>
                      </div>
                      <div className="w-full h-2 bg-muted rounded-full overflow-hidden mt-1">
                        <div
                          className={`h-full ${pct && pct > 100 ? "bg-red-500" : "bg-primary"}`}
                          style={{ width: `${Math.min(100, pct ?? 0)}%` }}
                        />
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Recommendations */}
          {/* <div className="border rounded-xl p-4 bg-card"> */}
          {/*   <div className="font-semibold text-card-foreground mb-4">Product Recommendations</div> */}
          {/*   {renderRecs()} */}
          {/* </div> */}

          <ProductGrid recommendations={recs} onAddToCartOptional={(item) =>{ console.log({item})} } />

          {/* Final summary */}
          <div ref={summaryRef} className="border rounded-xl p-4 bg-card">
            <div className="font-semibold text-card-foreground mb-4">AI Summary</div>
            {summary ? (
              <div className="prose max-w-none text-foreground whitespace-pre-wrap text-sm leading-relaxed">
                {summary}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">Waiting for summary…</div>
            )}
          </div>

          {/* Raw JSON toggle (handy while iterating) */}
          <div className="border rounded-xl p-4 bg-card">
            <button
              onClick={() => setShowRaw((s) => !s)}
              className="text-sm inline-flex items-center gap-1 text-muted-foreground hover:text-foreground"
            >
              {showRaw ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              {showRaw ? "Hide raw state" : "Show raw state"}
            </button>
            {showRaw && (
              <pre className="mt-3 text-xs overflow-auto max-h-64 bg-background p-3 rounded-lg border">
                {JSON.stringify(lastState, null, 2)}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

