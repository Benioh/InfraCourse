import Link from "next/link";
import { api } from "@/lib/api";
import { Panel } from "@/components/ui";
import { inferRepoPath, parseSourceHref, sourceUrl, splitInspectPaths } from "@/lib/source";

const frameworks = ["MiniInfra", "pytorch", "TorchTitan", "Megatron", "vLLM", "SGLang", "verl", "SLiME"];

function InspectLinks({ raw }: { raw: string }) {
  const parts = splitInspectPaths(raw);
  if (!parts.length) return <span>{raw}</span>;
  return (
    <span className="flex flex-wrap gap-2">
      {parts.map((part, idx) => {
        const { path, hitRange } = parseSourceHref(part);
        const ok = inferRepoPath(path);
        if (!ok) {
          return (
            <span key={`${part}-${idx}`} className="font-mono text-xs text-quest-muted">
              {part}
            </span>
          );
        }
        return (
          <Link
            key={`${part}-${idx}`}
            href={sourceUrl(path, hitRange)}
            className="rounded-full border border-quest-accent/70 bg-white/80 px-3 py-1 font-mono text-xs text-quest-accent hover:bg-quest-accent hover:text-white"
          >
            {part}
          </Link>
        );
      })}
    </span>
  );
}

export default async function SourceMapPage({ params }: { params: Promise<{ framework: string }> }) {
  const { framework } = await params;
  const payload = await api.sourceMap(framework);

  return (
    <div className="space-y-6">
      <Panel title={`${payload.framework} 源码地图`} eyebrow="Code Reading / 读代码路线">
        <div className="flex flex-wrap gap-2">
          {frameworks.map((item) => (
            <Link key={item} href={`/source-map/${item}`} className="rounded-full border border-quest-border bg-white/80 px-4 py-2 text-sm hover:border-quest-accent">
              {item}
            </Link>
          ))}
        </div>
      </Panel>
      <div className="grid gap-4">
        {payload.nodes.map((node) => (
          <Panel key={node.node} title={node.node} eyebrow="Inspection Path / 检查路径">
            <p className="text-sm text-quest-muted"><span className="font-semibold text-quest-ink">搜索：</span>{node.what_to_search}</p>
            <div className="mt-3 flex flex-col gap-2 text-sm text-quest-muted lg:flex-row lg:items-start lg:gap-3">
              <span className="font-semibold text-quest-ink">查看：</span>
              <InspectLinks raw={node.what_to_inspect} />
            </div>
            <p className="mt-3 text-sm"><span className="font-semibold">要回答的问题：</span>{node.question}</p>
          </Panel>
        ))}
      </div>
    </div>
  );
}
