"use client";
import { useState } from "react";
import { simulate, Diagnostics } from "@/lib/api";

const CHANNELS = ["google", "microsoft", "meta"];

export default function BudgetSimulator() {
  const [budget, setBudget] = useState<Record<string, number>>({
    google: 100,
    microsoft: 100,
    meta: 100,
  });
  const [diag, setDiag] = useState<Diagnostics | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try {
      setDiag((await simulate(30, budget)).diagnostics);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-lg border bg-white p-4">
      <h2 className="text-lg font-medium">3 · Budget simulator</h2>
      {CHANNELS.map((c) => (
        <div key={c} className="mt-2 flex items-center gap-3">
          <span className="w-24 text-sm capitalize">{c}</span>
          <input
            type="range"
            min={0}
            max={1000}
            value={budget[c]}
            onChange={(e) => setBudget({ ...budget, [c]: Number(e.target.value) })}
          />
          <span className="w-20 text-sm">${budget[c]}/day</span>
        </div>
      ))}
      <button
        onClick={run}
        className="mt-3 rounded bg-blue-600 px-3 py-1 text-sm text-white"
      >
        {busy ? "Re-forecasting…" : "Re-forecast"}
      </button>
      {diag && (
        <p className="mt-3 text-sm">
          Revenue P50: base ${diag.scenario.revenue_p50_base?.toFixed(0)} → scenario $
          {diag.scenario.revenue_p50_scenario?.toFixed(0)}
        </p>
      )}
    </section>
  );
}
