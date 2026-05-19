import Link from "next/link";
import { api } from "@/lib/api";
import { Panel } from "@/components/ui";
import { withPublicPrefix } from "@/lib/runtime";

export default async function TicketsPage() {
  const tickets = await api.tickets();

  return (
    <Panel title="Infra Debugger" eyebrow="Decision Trees / 排障决策树">
      <div className="grid gap-4 md:grid-cols-2">
        {tickets.map((ticket) => (
          <Link key={ticket.id} href={withPublicPrefix(`/tickets/${ticket.id}`)} className="rounded-3xl border border-quest-border bg-white/70 p-5 transition hover:border-quest-accent">
            <div className="font-mono text-sm text-quest-accent">{ticket.id}</div>
            <h3 className="mt-3 text-lg font-semibold">{ticket.title}</h3>
            <p className="mt-2 text-sm text-quest-muted">任务：{ticket.mission} · 严重程度：{ticket.severity}</p>
            <p className="mt-3 text-sm">{ticket.symptom}</p>
          </Link>
        ))}
      </div>
    </Panel>
  );
}
