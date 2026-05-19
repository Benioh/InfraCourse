import Link from "next/link";
import { api } from "@/lib/api";
import { CommandBlock, KeyValue, Panel, StatusPill } from "@/components/ui";
import { withPublicPrefix } from "@/lib/runtime";

export default async function CurriculumPage() {
  const curriculum = await api.curriculum();
  const current = curriculum.current_mission;

  return (
    <div className="space-y-6">
      <Panel title={curriculum.title} eyebrow="Self Study / 单人自学路线">
        <p className="max-w-4xl text-sm leading-6 text-quest-muted">{curriculum.principle}</p>
        <div className="mt-5 grid gap-4 md:grid-cols-5">
          <KeyValue
            label="关卡"
            value={`${curriculum.stats.completed_missions}/${curriculum.stats.total_missions}`}
          />
          <KeyValue label="源码节点" value={curriculum.stats.total_source_nodes} />
          <KeyValue label="Notebook" value={curriculum.stats.total_notebooks} />
          <KeyValue label="Patch 任务" value={curriculum.stats.total_projects} />
          <KeyValue label="当前建议" value={current?.level ?? "—"} />
        </div>
      </Panel>

      {current ? (
        <Panel
          title={`下一步 · ${current.level} ${current.title}`}
          eyebrow={current.role ?? "Next Patch"}
        >
          <div className="grid gap-4 lg:grid-cols-[1fr_1.2fr]">
            <div className="space-y-3 text-sm text-quest-muted">
              {current.patch_description ? (
                <p className="whitespace-pre-wrap">{current.patch_description}</p>
              ) : (
                <p>本关 Patch 任务尚未配置描述。</p>
              )}
              <p>
                源码 {current.source_count} · Notebook {current.notebook_count} · Ticket{" "}
                {current.ticket_count}
                {current.patch_test_count
                  ? ` · ${current.patch_test_count} 个测试`
                  : ""}
              </p>
              <Link
                href={withPublicPrefix(`/missions/${current.id}`)}
                className="inline-flex rounded-full border border-quest-accent px-4 py-2 font-medium text-quest-accent hover:bg-quest-accent hover:text-white"
              >
                打开本关任务
              </Link>
            </div>
            {current.recommended_command ? (
              <CommandBlock command={current.recommended_command} />
            ) : null}
          </div>
        </Panel>
      ) : null}

      <Panel title="单人学习闭环" eyebrow="Loop / 每关固定节奏">
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {curriculum.self_study_loop.map((step, index) => (
            <article
              key={step.name}
              className="rounded-2xl border border-quest-border/70 bg-white/70 p-4"
            >
              <div className="flex items-center justify-between gap-3">
                <h3 className="font-semibold">
                  {index + 1}. {step.name}
                </h3>
                <span className="rounded-full bg-[#f8f4ec] px-3 py-1 text-xs text-quest-muted">
                  {step.duration}
                </span>
              </div>
              <p className="mt-3 text-sm leading-6 text-quest-muted">{step.action}</p>
            </article>
          ))}
        </div>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="卡住时怎么办" eyebrow="Rescue / 自救路径">
          <div className="space-y-3">
            {curriculum.stuck_playbook.map((item) => (
              <div
                key={item.symptom}
                className="rounded-2xl border border-quest-border/70 bg-white/70 p-4"
              >
                <p className="font-semibold">{item.symptom}</p>
                <p className="mt-2 text-sm leading-6 text-quest-muted">{item.action}</p>
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="每关完成检查" eyebrow="Checklist / 不靠感觉过关">
          <ul className="space-y-3 text-sm text-quest-muted">
            {curriculum.completion_checks.map((check) => (
              <li
                key={check}
                className="rounded-2xl border border-quest-border/70 bg-white/70 p-3"
              >
                ✓ {check}
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <Panel title="路线阶段" eyebrow="Roadmap / 全部 lab">
        <div className="space-y-5">
          {curriculum.phases.map((phase) => (
            <section
              key={phase.name}
              className="rounded-3xl border border-quest-border/70 bg-white/70 p-5"
            >
              <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <h3 className="font-mono text-lg font-semibold">{phase.name}</h3>
                  <p className="mt-2 text-sm text-quest-muted">
                    框架：{phase.frameworks.join(", ") || "—"} · 源码节点{" "}
                    {phase.source_count} · Notebook {phase.notebook_count}
                  </p>
                </div>
                <span className="rounded-full border border-quest-border px-3 py-1 text-sm text-quest-muted">
                  {phase.completed_missions}/{phase.missions.length} 已通过
                </span>
              </div>
              <div className="mt-4 grid gap-3 lg:grid-cols-2">
                {phase.missions.map((mission) => (
                  <Link
                    key={mission.id}
                    href={withPublicPrefix(`/missions/${mission.id}`)}
                    className="rounded-2xl border border-quest-border bg-[#fbfaf7] p-4 transition hover:border-quest-accent hover:shadow-panel"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="font-mono text-sm text-quest-muted">{mission.level}</p>
                        <h4 className="mt-1 font-semibold">{mission.title}</h4>
                      </div>
                      <StatusPill
                        passed={mission.status === "已通过"}
                        label={mission.status}
                      />
                    </div>
                    {mission.patch_description ? (
                      <p className="mt-3 line-clamp-2 text-sm text-quest-muted">
                        {mission.patch_description}
                      </p>
                    ) : null}
                    <p className="mt-2 text-xs text-quest-muted">
                      源码 {mission.source_count} · Notebook {mission.notebook_count} ·
                      Ticket {mission.ticket_count}
                      {mission.patch_test_count
                        ? ` · ${mission.patch_test_count} 测试`
                        : ""}
                    </p>
                  </Link>
                ))}
              </div>
            </section>
          ))}
        </div>
      </Panel>

      <Panel title="自学文档入口" eyebrow="Docs / 先读这些">
        <div className="grid gap-4 lg:grid-cols-2">
          {curriculum.docs.map((doc) => (
            <article
              key={doc.path}
              className="rounded-2xl border border-quest-border/70 bg-white/70 p-4"
            >
              <p className="font-semibold">{doc.title}</p>
              <p className="mt-2 font-mono text-xs text-quest-muted">{doc.path}</p>
              <p className="mt-3 text-sm leading-6 text-quest-muted">{doc.summary}</p>
            </article>
          ))}
        </div>
      </Panel>
    </div>
  );
}
