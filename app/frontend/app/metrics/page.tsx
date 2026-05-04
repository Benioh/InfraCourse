import { api } from "@/lib/api";
import { MetricChart } from "@/components/metric-chart";
import { Panel } from "@/components/ui";

function normalizeSeries(rows: Record<string, unknown>[], key: string) {
  return rows
    .filter((row) => typeof row[key] === "number")
    .map((row, index) => ({ x: String(row.step ?? index), [key]: row[key] as number }));
}

export default async function MetricsPage() {
  const metrics = await api.metrics();
  const lossSeries = normalizeSeries(metrics, "loss");
  const throughputSeries = normalizeSeries(metrics, "tokens_per_sec");
  const ttftSeries = normalizeSeries(metrics, "ttft_ms_p50");

  return (
    <div className="space-y-6">
      <Panel title="指标看板" eyebrow="Runs Contract / 运行产物约定">
        <p className="text-sm text-quest-muted">
          指标来自 <code>runs/{"{mission_id}"}/{"{run_id}"}/metrics.jsonl</code>，后端会统一读取并展示。
        </p>
      </Panel>
      <div className="grid gap-6 xl:grid-cols-2">
        <MetricChart data={lossSeries} dataKey="loss" title="Loss 曲线" color="#b45309" />
        <MetricChart data={throughputSeries} dataKey="tokens_per_sec" title="Tokens / Sec" color="#0f766e" />
      </div>
      <div className="grid gap-6 xl:grid-cols-2">
        <MetricChart data={ttftSeries} dataKey="ttft_ms_p50" title="TTFT p50" color="#7c3aed" />
        <Panel title="最近原始指标" eyebrow="JSONL">
          <pre className="max-h-[28rem] overflow-auto whitespace-pre-wrap rounded-2xl border border-quest-border bg-[#201d18] p-4 text-xs text-[#f5ecdf]">
            {JSON.stringify(metrics.slice(-20), null, 2)}
          </pre>
        </Panel>
      </div>
    </div>
  );
}
