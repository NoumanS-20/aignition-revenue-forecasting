"use client";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export type FanPoint = { label: string; p10: number; p50: number; p90: number };

export default function FanChart({ data }: { data: FanPoint[] }) {
  const shaped = data.map((d) => ({ ...d, band: [d.p10, d.p90] as [number, number] }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={shaped}>
        <XAxis dataKey="label" />
        <YAxis />
        <Tooltip />
        <Area dataKey="band" stroke="none" fill="#bfdbfe" />
        <Line dataKey="p50" stroke="#2563eb" dot={false} strokeWidth={2} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
