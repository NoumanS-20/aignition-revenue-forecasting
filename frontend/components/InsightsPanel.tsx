"use client";
import { useState } from "react";
import { getInsights } from "@/lib/api";

type Insights = { narrative: string; risks: string[]; recommendations: string[] };

export default function InsightsPanel() {
  const [data, setData] = useState<Insights | null>(null);
  const [busy, setBusy] = useState(false);

  async function explain() {
    setBusy(true);
    try {
      setData(await getInsights(30));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-lg border bg-white p-4">
      <h2 className="text-lg font-medium">4 · AI insights</h2>
      <button
        onClick={explain}
        className="mt-2 rounded bg-blue-600 px-3 py-1 text-sm text-white"
      >
        {busy ? "Thinking…" : "Explain this forecast"}
      </button>
      {data && (
        <div className="mt-3 space-y-2 text-sm">
          <p>{data.narrative}</p>
          {data.risks.length > 0 && (
            <div>
              <b>Risks</b>
              <ul className="list-disc pl-5">
                {data.risks.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
          {data.recommendations.length > 0 && (
            <div>
              <b>Recommendations</b>
              <ul className="list-disc pl-5">
                {data.recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
