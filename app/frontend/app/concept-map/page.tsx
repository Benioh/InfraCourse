import { api } from "@/lib/api";
import { Panel } from "@/components/ui";

export default async function ConceptMapPage() {
  const nodes = await api.conceptMap();

  return (
    <Panel title="概念图" eyebrow="Systems Vocabulary / 系统词汇表">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {nodes.map((node) => (
          <article key={node.id} className="rounded-3xl border border-quest-border bg-white/70 p-5">
            <div className="font-mono text-sm text-quest-accent">{node.id}</div>
            <p className="mt-3 text-sm text-quest-muted">{node.definition}</p>
            <p className="mt-3 text-sm"><span className="font-semibold">为什么重要：</span>{node.why_it_matters}</p>
            <p className="mt-3 text-sm"><span className="font-semibold">出现位置：</span>{node.where_it_appears.join(", ")}</p>
            <p className="mt-3 text-sm"><span className="font-semibold">关联实验：</span>{node.related_experiments.join(", ")}</p>
            <p className="mt-3 text-sm text-quest-danger"><span className="font-semibold">常见失败：</span>{node.common_failure}</p>
          </article>
        ))}
      </div>
    </Panel>
  );
}
