import { api } from "@/lib/api";
import { Panel } from "@/components/ui";

export default async function TicketDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const ticket = await api.ticket(id);

  return (
    <div className="space-y-6">
      <Panel title={ticket.title} eyebrow={ticket.id}>
        <p className="text-sm text-quest-muted">任务：{ticket.mission} · 严重程度：{ticket.severity}</p>
        <p className="mt-4 text-sm">{ticket.symptom}</p>
        {ticket.error_signatures && ticket.error_signatures.length > 0 ? (
          <div className="mt-4 rounded-lg border border-quest-border bg-slate-50/70 p-3 font-mono text-xs text-slate-700">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-quest-muted">典型错误签名</p>
            <ul className="space-y-1">
              {ticket.error_signatures.map((sig) => (
                <li key={sig}>• {sig}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </Panel>

      {(ticket.hint_1 || ticket.hint_2 || ticket.solution) ? (
        <Panel title="渐进揭示" eyebrow="Hint-1 / Hint-2 / 解答">
          <p className="mb-4 text-xs text-quest-muted">
            建议先只打开 Hint-1，按指引在自己环境跑一次最小检查再看下一层。直接跳到「解答」会把可训练的排障直觉喂掉。
          </p>
          <div className="space-y-3">
            {ticket.hint_1 ? (
              <details className="group rounded-lg border border-quest-border bg-white/70 p-4 transition open:border-quest-accent">
                <summary className="cursor-pointer select-none text-sm font-semibold text-quest-ink">
                  Hint-1 · 先做这一步
                </summary>
                <div className="mt-3 whitespace-pre-wrap text-sm text-quest-muted">{ticket.hint_1}</div>
              </details>
            ) : null}
            {ticket.hint_2 ? (
              <details className="group rounded-lg border border-quest-border bg-white/70 p-4 transition open:border-quest-accent">
                <summary className="cursor-pointer select-none text-sm font-semibold text-quest-ink">
                  Hint-2 · 仍未定位时再展开
                </summary>
                <div className="mt-3 whitespace-pre-wrap text-sm text-quest-muted">{ticket.hint_2}</div>
              </details>
            ) : null}
            {ticket.solution ? (
              <details className="group rounded-lg border border-amber-200 bg-amber-50/60 p-4 transition open:border-amber-400">
                <summary className="cursor-pointer select-none text-sm font-semibold text-amber-800">
                  解答 · 根因与修复
                </summary>
                <div className="mt-3 whitespace-pre-wrap text-sm text-amber-900">{ticket.solution}</div>
              </details>
            ) : null}
          </div>
        </Panel>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="分类问题" eyebrow="Triage / 定位阶段"><ul className="space-y-2 text-sm text-quest-muted">{(ticket.classification_questions ?? []).map((item) => <li key={item}>• {item}</li>)}</ul></Panel>
        <Panel title="可能原因" eyebrow="Hypotheses / 假设"><ul className="space-y-2 text-sm text-quest-muted">{(ticket.likely_causes ?? []).map((item) => <li key={item}>• {item}</li>)}</ul></Panel>
        <Panel title="最小检查" eyebrow="Observe / 先观察"><ul className="space-y-2 text-sm text-quest-muted">{(ticket.minimal_checks ?? []).map((item) => <li key={item}>• {item}</li>)}</ul></Panel>
        <Panel title="最小修复" eyebrow="Patch Plan / 补丁计划"><ul className="space-y-2 text-sm text-quest-muted">{(ticket.minimal_fixes ?? []).map((item) => <li key={item}>• {item}</li>)}</ul></Panel>
      </div>
      <Panel title="不要这样做" eyebrow="Guardrails / 护栏">
        <ul className="space-y-2 text-sm text-quest-danger">{(ticket.what_not_to_do ?? []).map((item) => <li key={item}>• {item}</li>)}</ul>
      </Panel>
    </div>
  );
}
