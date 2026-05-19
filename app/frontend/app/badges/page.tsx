import { api } from "@/lib/api";
import { Panel, StatusPill } from "@/components/ui";
import Link from "next/link";
import { withPublicPrefix } from "@/lib/runtime";

export default async function BadgesPage() {
  const badges = await api.badges();

  return (
    <Panel title="徽章进度" eyebrow="Patch PASS = 徽章解锁">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {badges.map((row) => (
          <Link
            key={row.mission}
            href={withPublicPrefix(`/missions/${row.mission}`)}
            className="rounded-3xl border border-quest-border bg-white/70 p-5 transition hover:border-quest-accent hover:shadow-panel"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="font-mono text-sm text-quest-muted">{row.level ?? "—"}</span>
              <StatusPill passed={row.passed} label={row.passed ? "已通过" : "未完成"} />
            </div>
            <h3 className="mt-4 text-lg font-semibold">{row.title ?? row.mission}</h3>
            <p className="mt-2 font-mono text-xs text-quest-muted">{row.mission}</p>
            <p className="mt-3 text-sm">
              {row.summary
                ? `测试 ${row.summary.passed}/${row.summary.total} 通过`
                : "Patch 还未运行"}
              {row.test_count ? ` · 共 ${row.test_count} 个测试` : ""}
            </p>
            {row.finished_at ? (
              <p className="mt-1 text-xs text-quest-muted">
                最近运行：{new Date(row.finished_at).toLocaleString()}
              </p>
            ) : null}
          </Link>
        ))}
      </div>
    </Panel>
  );
}
