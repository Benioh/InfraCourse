import Link from "next/link";
import { api } from "@/lib/api";
import { CommandBlock, KeyValue, Panel, StatusPill } from "@/components/ui";

function statusLabel(status: string): { label: string; passed: boolean } {
  if (status === "已通过") return { label: "已通过", passed: true };
  if (status === "失败重试") return { label: "失败重试", passed: false };
  return { label: "未开始", passed: false };
}

export default async function DashboardPage() {
  const dashboard = await api.dashboard();
  const active = dashboard.missions.find((m) => m.id === dashboard.active_mission);
  const recommendedCmd =
    active?.patch_test_command ?? "make patch-test M=l02_pytorch_systems";

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-4">
        <KeyValue label="当前关卡" value={dashboard.current_level ?? "L01"} />
        <KeyValue label="当前角色" value={dashboard.current_role ?? "Infra 新兵"} />
        <KeyValue label="当前任务" value={dashboard.active_mission ?? "—"} />
        <KeyValue
          label="完成进度"
          value={`${dashboard.completed_missions}/${dashboard.total_missions}`}
        />
      </div>

      <Panel title="推荐下一条命令" eyebrow="Patch Track / 写补丁 + 跑测试">
        <CommandBlock command={recommendedCmd} />
        <p className="mt-2 text-sm text-quest-muted">
          每关只做一件事：改 <code>patch/starter/</code> 里的代码，跑过测试就过关。
        </p>
      </Panel>

      <Panel title="任务队列" eyebrow="按章节顺序学习">
        <div className="grid gap-4">
          {dashboard.missions.map((mission) => {
            const status = statusLabel(mission.status);
            const summary = mission.patch_status?.summary;
            const lastRun = mission.patch_status?.finished_at;
            return (
              <Link
                key={mission.id}
                href={`/missions/${mission.id}`}
                className="rounded-3xl border border-quest-border bg-white/70 p-5 transition hover:border-quest-accent hover:shadow-panel"
              >
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-sm text-quest-muted">
                        {mission.level}
                      </span>
                      <StatusPill passed={status.passed} label={status.label} />
                      {mission.no_gpu_friendly ? (
                        <span className="rounded-full border border-quest-border px-2 py-0.5 text-xs text-quest-muted">
                          0 GPU
                        </span>
                      ) : null}
                      {mission.estimated_minutes ? (
                        <span className="rounded-full border border-quest-border px-2 py-0.5 text-xs text-quest-muted">
                          ~{mission.estimated_minutes}min
                        </span>
                      ) : null}
                    </div>
                    <h3 className="mt-3 text-xl font-semibold">{mission.title}</h3>
                    <p className="mt-2 text-sm text-quest-muted">
                      框架：{(mission.frameworks ?? []).join(", ") || "—"}
                    </p>
                  </div>
                  <div className="min-w-64">
                    <div className="rounded-2xl border border-quest-border/70 bg-[#f8f4ec] p-4">
                      <div className="text-xs uppercase tracking-[0.22em] text-quest-muted">
                        Patch 状态
                      </div>
                      {summary ? (
                        <div className="mt-2 font-mono text-sm">
                          {summary.passed}/{summary.total} 通过
                        </div>
                      ) : (
                        <div className="mt-2 font-mono text-sm text-quest-muted">
                          从未运行
                        </div>
                      )}
                      {lastRun ? (
                        <div className="mt-2 text-xs text-quest-muted">
                          {new Date(lastRun).toLocaleString()}
                        </div>
                      ) : null}
                      <div className="mt-3 text-xs text-quest-muted">
                        {mission.patch_test_command ?? "尚未配置 Patch Track"}
                      </div>
                    </div>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}
