"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function MetricChart({
  data,
  dataKey,
  title,
  color = "#0f766e",
}: {
  data: Record<string, unknown>[];
  dataKey: string;
  title: string;
  color?: string;
}) {
  return (
    <div className="rounded-panel border border-quest-border bg-quest-card p-5 shadow-panel">
      <h3 className="font-mono text-lg font-semibold">{title}</h3>
      <div className="mt-4 h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#e5ddd2" strokeDasharray="4 4" />
            <XAxis dataKey="x" tick={{ fill: "#6d655d", fontSize: 12 }} />
            <YAxis tick={{ fill: "#6d655d", fontSize: 12 }} />
            <Tooltip />
            <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
