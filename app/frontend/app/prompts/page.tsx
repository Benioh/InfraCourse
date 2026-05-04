import { api } from "@/lib/api";
import { Panel } from "@/components/ui";

export default async function PromptsPage() {
  const prompts = await api.prompts();

  return (
    <div className="space-y-6">
      <Panel title="AI 协作提示卡" eyebrow="Codex / Cursor">
        <p className="text-sm text-quest-muted">
          标准协议：Observe → Ask → Patch Plan → Human Check → Apply → Test → Explain → Commit。AI 是 copilot，不是 oracle。
        </p>
      </Panel>
      <div className="grid gap-4">
        {prompts.map((prompt) => (
          <Panel key={prompt.id} title={prompt.title} eyebrow={prompt.path}>
            <pre className="whitespace-pre-wrap rounded-2xl border border-quest-border bg-[#201d18] p-4 text-sm text-[#f5ecdf]">
              {prompt.body}
            </pre>
          </Panel>
        ))}
      </div>
    </div>
  );
}
