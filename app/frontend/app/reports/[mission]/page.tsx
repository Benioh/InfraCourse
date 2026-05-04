import { api } from "@/lib/api";
import { Panel } from "@/components/ui";

export default async function ReportPage({ params }: { params: Promise<{ mission: string }> }) {
  const { mission } = await params;
  const report = await api.report(mission);

  return (
    <Panel title={`任务报告 · ${mission}`} eyebrow={report.source ?? "未找到 report.md"}>
      <article className="whitespace-pre-wrap rounded-3xl border border-quest-border bg-white/70 p-5 font-mono text-sm leading-7">
        {report.body}
      </article>
    </Panel>
  );
}
