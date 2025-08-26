import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * RecommendStreamDemo
 *
 * A single-file React component that:
 *  - renders a textbox to enter a shopping goal
 *  - connects to the FastAPI SSE endpoint `/recommend/stream?user_goal=...`
 *  - streams node-by-node updates and shows progress in real-time
 *  - displays categories, pricing/decisions (lite view), and the final summary
 *
 * TailwindCSS classes are used for styling. No other deps are required.
 *
 * Usage:
 *   import RecommendStreamDemo from "./RecommendStreamDemo";
 *   export default function App() { return <RecommendStreamDemo /> }
 */

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

type PublicSlice = {
  category_plan?: string[];
  spec?: Record<string, any>;
  recommendations?: Record<string, any>;
  final_output?: string | null;
};

type NodeEndEvent = {
  node: string;
  state: PublicSlice;
  ts: number;
};

type StartEvent = { user_goal: string; ts: number };

type DoneEvent = { state: PublicSlice; ts: number };

export default function RecommendStreamDemo() {
  const [goal, setGoal] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [categories, setCategories] = useState<string[]>([]);
  const [spec, setSpec] = useState<Record<string, any>>({});
  const [recs, setRecs] = useState<Record<string, any>>({});
  const [summary, setSummary] = useState<string>("");
  const [lastNode, setLastNode] = useState<string | null>(null);
  const [errors, setErrors] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<Record<string, "pending" | "done" | "active">>({});

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

  const resetUI = () => {
    setCategories([]);
    setSpec({});
    setRecs({});
    setSummary("");
    setLastNode(null);
    setErrors(null);
    setTimeline(initialTimeline);
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
      // could show a toast / log here
    });

    es.addEventListener("node_end", (evt) => {
      const { node, state } = JSON.parse((evt as MessageEvent).data) as NodeEndEvent;
      setLastNode(node);

      // progress timeline: mark current as done, advance next to active
      setTimeline((prev) => {
        const next = { ...prev };
        if (node in next) next[node] = "done";
        // advance the next known step to active
        const idx = KNOWN_STEPS.findIndex((s) => s.id === node);
        if (idx >= 0 && idx + 1 < KNOWN_STEPS.length) {
          const nextId = KNOWN_STEPS[idx + 1].id;
          if (next[nextId] === "pending") next[nextId] = "active";
        }
        return next;
      });

      if (state.category_plan) setCategories(state.category_plan);
      if (state.spec) setSpec(state.spec);
      if (state.recommendations) setRecs(state.recommendations);
      if (state.final_output) setSummary(state.final_output || "");
    });

    es.addEventListener("done", (evt) => {
      const data = JSON.parse((evt as MessageEvent).data) as DoneEvent;
      if (data.state?.final_output) setSummary(data.state.final_output);
      setTimeline((prev) => {
        const next = { ...prev };
        KNOWN_STEPS.forEach((s) => (next[s.id] = next[s.id] === "pending" ? "done" : next[s.id]));
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
      es.close();
      esRef.current = null;
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
    const base = "px-2 py-0.5 text-xs rounded-full";
    if (status === "done") return <span className={`${base} bg-green-100 text-green-700`}>done</span>;
    if (status === "active") return <span className={`${base} bg-blue-100 text-blue-700`}>running</span>;
    return <span className={`${base} bg-gray-100 text-gray-600`}>pending</span>;
  };

  const renderRecs = () => {
    const keys = Object.keys(recs || {});
    if (!keys.length) return <div className="text-sm text-gray-500">Waiting for retrieval…</div>;
    return (
      <div className="space-y-4">
        {keys.map((cat) => {
          const val = recs[cat];
          const items = Array.isArray(val?.top_comments) ? val.top_comments : [];
          return (
            <div key={cat} className="border rounded-2xl p-4 shadow-sm">
              <div className="font-semibold mb-2">{cat}</div>
              {!items.length ? (
                <div className="text-sm text-gray-500">No items yet…</div>
              ) : (
                <ul className="space-y-2">
                  {items.map((c: any, idx: number) => (
                    <li key={idx} className="flex flex-col md:flex-row md:items-center md:justify-between gap-2 border rounded-xl p-3">
                      <div>
                        <div className="text-sm font-medium">{c.product}</div>
                        {typeof c.votes === "number" && (
                          <div className="text-xs text-gray-500">votes: {c.votes}</div>
                        )}
                      </div>
                      <div className="flex items-center gap-3">
                        {c.best_price != null && (
                          <span className="text-sm font-semibold">{c.currency || "USD"} {Number(c.best_price).toFixed(2)}</span>
                        )}
                        {c.agent_decision && (
                          <span className="text-xs px-2 py-1 rounded-full bg-amber-100 text-amber-800">{c.agent_decision}</span>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="mx-auto max-w-4xl p-6 space-y-6">
      <h1 className="text-2xl font-bold">Smart Shopping (streaming demo)</h1>

      <div className="grid gap-3 md:grid-cols-[1fr_auto] items-center">
        <input
          className="w-full border rounded-xl px-4 py-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="e.g. I want a budget home espresso setup under $500"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          disabled={isStreaming}
        />
        <div className="flex gap-2">
          {!isStreaming ? (
            <button
              onClick={start}
              className="px-4 py-3 rounded-xl bg-blue-600 text-white shadow hover:bg-blue-700"
            >
              Start
            </button>
          ) : (
            <button
              onClick={stop}
              className="px-4 py-3 rounded-xl bg-gray-200 text-gray-800 hover:bg-gray-300"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {errors && (
        <div className="rounded-xl bg-red-50 text-red-700 p-3 text-sm">{errors}</div>
      )}

      {/* Progress timeline */}
      <div className="border rounded-2xl p-4 shadow-sm">
        <div className="font-semibold mb-3">Progress</div>
        <ol className="space-y-2">
          {KNOWN_STEPS.map((s) => (
            <li key={s.id} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full"
                  style={{ background: timeline[s.id] === "done" ? "#16a34a" : timeline[s.id] === "active" ? "#2563eb" : "#9ca3af" }}
                />
                <span>{s.label}</span>
              </div>
              <StepBadge status={timeline[s.id] || "pending"} />
            </li>
          ))}
        </ol>
      </div>

      {/* Categories & Spec */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="border rounded-2xl p-4 shadow-sm">
          <div className="font-semibold mb-2">Categories</div>
          {!categories.length ? (
            <div className="text-sm text-gray-500">Waiting for goal analysis…</div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {categories.map((c) => (
                <span key={c} className="text-sm px-2 py-1 rounded-full bg-gray-100">{c}</span>
              ))}
            </div>
          )}
        </div>
        <div className="border rounded-2xl p-4 shadow-sm">
          <div className="font-semibold mb-2">Spec</div>
          {!Object.keys(spec).length ? (
            <div className="text-sm text-gray-500">Waiting for spec builder…</div>
          ) : (
            <pre className="text-xs bg-gray-50 p-3 rounded-xl overflow-auto">{JSON.stringify(spec, null, 2)}</pre>
          )}
        </div>
      </div>

      {/* Recommendations (lite) */}
      <div className="border rounded-2xl p-4 shadow-sm">
        <div className="font-semibold mb-2">Recommendations</div>
        {renderRecs()}
      </div>

      {/* Final summary */}
      <div ref={summaryRef} className="border rounded-2xl p-4 shadow-sm">
        <div className="font-semibold mb-2">Summary</div>
        {summary ? (
          <div className="prose max-w-none whitespace-pre-wrap">{summary}</div>
        ) : (
          <div className="text-sm text-gray-500">Waiting for summary…</div>
        )}
      </div>
    </div>
  );
}
