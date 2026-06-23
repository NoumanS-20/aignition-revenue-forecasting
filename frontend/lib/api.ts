const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type ForecastRow = {
  level: string;
  entity: string;
  metric: "revenue" | "roas";
  horizon_days: number;
  p10: number;
  p50: number;
  p90: number;
};

export type Diagnostics = {
  horizon_days: number;
  total_revenue_p50: number;
  blended_roas_p50: number;
  scenario: {
    applied: boolean;
    revenue_p50_base: number;
    revenue_p50_scenario: number;
  };
  series: {
    channel: string;
    campaign: string;
    recent_spend: number;
    half_saturation: number;
    saturation: string;
  }[];
};

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return r.json();
}

export const getForecast = (horizon: number) =>
  post<{ rows: ForecastRow[] }>("/forecast", { horizon });

export const simulate = (horizon: number, budget_plan: Record<string, number>) =>
  post<{ rows: ForecastRow[]; diagnostics: Diagnostics }>("/simulate", {
    horizon,
    budget_plan,
  });

export const getInsights = (horizon: number, budget_plan?: Record<string, number>) =>
  post<{ narrative: string; risks: string[]; recommendations: string[] }>(
    "/insights",
    { horizon, budget_plan }
  );

export async function validate() {
  const r = await fetch(`${BASE}/validate`);
  return r.json();
}

export async function uploadFiles(files: File[]) {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  const r = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
  return r.json();
}
