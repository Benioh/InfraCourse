import Link from "next/link";
import { api } from "@/lib/api";
import { CommandBlock, KeyValue, Panel, StatusPill } from "@/components/ui";
import { withPublicPrefix } from "@/lib/runtime";

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

      <Panel title="先听一小节课，再做一个小补丁" eyebrow="Lesson First / 先上课，再做题">
        <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="text-sm leading-7 text-quest-ink">
            <p>每关都像一篇教学博客：先讲背景，再穿插源码，最后用 Quiz 和 Patch 检查你是否真的懂。</p>
            <p className="mt-2 text-quest-muted">不要一进来就跑命令或读完整源码。先进入本关讲义，跟着高亮片段看真实实现。</p>
            {active ? (
              <Link
                href={withPublicPrefix(`/missions/${active.id}#lesson`)}
                className="mt-4 inline-flex rounded-full bg-quest-accent px-4 py-2 font-medium text-white hover:opacity-90"
              >
                进入本关讲义
              </Link>
            ) : null}
          </div>
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-eyebrow text-quest-muted">最后验证命令</p>
            <CommandBlock command={recommendedCmd} />
            <p className="mt-2 text-sm text-quest-muted">读完讲义、过 Quiz、写完 Lab 后，再用它确认实现正确。</p>
          </div>
        </div>
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
                href={withPublicPrefix(`/missions/${mission.id}`)}
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
                        {mission.id === dashboard.active_mission ? "当前建议：先进入本关讲义" : (mission.patch_test_command ?? "进入讲义学习")}
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
