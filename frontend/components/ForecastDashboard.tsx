"use client";
import { useEffect, useState } from "react";
import { getForecast, ForecastRow } from "@/lib/api";
import FanChart, { FanPoint } from "./FanChart";

const HORIZONS = [30, 60, 90];

export default function ForecastDashboard() {
  const [horizon, setHorizon] = useState(30);
  const [rows, setRows] = useState<ForecastRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getForecast(horizon)
      .then((r) => {
        setRows(r.rows);
        setError(null);
      })
      .catch((e) => setError(String(e)));
  }, [horizon]);

  const channelRevenue: FanPoint[] = rows
    .filter((r) => r.level === "channel" && r.metric === "revenue")
    .map((r) => ({ label: r.entity, p10: r.p10, p50: r.p50, p90: r.p90 }));
  const totalRoas = rows.find((r) => r.level === "total" && r.metric === "roas");

  return (
    <section className="rounded-lg border bg-white p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">2 · Forecast</h2>
        <div className="flex gap-2">
          {HORIZONS.map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              className={`rounded px-3 py-1 text-sm ${
                h === horizon ? "bg-blue-600 text-white" : "bg-gray-100"
              }`}
            >
              {h}d
            </button>
          ))}
        </div>
      </div>
      {error && <p className="mt-2 text-sm text-red-700">Start the API: {error}</p>}
      {totalRoas && (
        <p className="mt-2 text-sm">
          Blended ROAS (P50): <b>{totalRoas.p50.toFixed(2)}</b> (P10{" "}
          {totalRoas.p10.toFixed(2)} – P90 {totalRoas.p90.toFixed(2)})
        </p>
      )}
      <FanChart data={channelRevenue} />
    </section>
  );
}
