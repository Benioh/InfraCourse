import Link from "next/link";
import { api } from "@/lib/api";
import { Panel } from "@/components/ui";
import PatchRunner from "@/components/patch-runner";
import QuizGate from "@/components/quiz-gate";
import { inferRepoPath, parseSourceHref, sourceUrl } from "@/lib/source";
import type { PatchPayload } from "@/lib/types";

function jupyterUrl(path: string) {
  const base = process.env.NEXT_PUBLIC_JUPYTER_BASE ?? "http://localhost:8888";
  return `${base.replace(/\/$/, "")}/lab/tree/${path}`;
}

function MetaBadge({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full border border-quest-border bg-white px-3 py-1 text-xs text-quest-muted">
      {children}
    </span>
  );
}

export default async function MissionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  // Fetch quest metadata + patch payload in parallel.
  const [quest, patch] = await Promise.all([
    api.quest(id),
    api.patch(id).catch<PatchPayload | null>(() => null),
  ]);

  // Graceful fallback if a lab hasn't been migrated to Patch Track yet.
  if (!patch) {
    return (
      <div className="space-y-4">
        <Panel title={`${quest.level ?? ""} · ${quest.title}`} eyebrow={quest.role}>
          <p className="text-sm text-quest-muted">
            本关尚未配置 Patch Track（quests/{id}.yaml 缺少 <code>patch:</code> 块）。
            目前只能阅读源码与 notebook，作者完成 patch 设计后这里会出现新的任务面板。
          </p>
        </Panel>
        <ResourceSidebar quest={quest} patch={null} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* ============================================================ */}
      {/*  Header                                                       */}
      {/* ============================================================ */}
      <Panel title={`${quest.level ?? ""} · ${quest.title}`} eyebrow={quest.role ?? "Patch Track"}>
        <div className="flex flex-wrap items-center gap-2 text-sm text-quest-muted">
          <MetaBadge>章节：{quest.act ?? "—"}</MetaBadge>
          <MetaBadge>预计 {quest.estimated_minutes ?? "?"} 分钟</MetaBadge>
          {quest.no_gpu_friendly ? <MetaBadge>0 GPU 可完成</MetaBadge> : null}
          {patch.test_count ? <MetaBadge>{patch.test_count} 个测试</MetaBadge> : null}
          {patch.test_kind ? <MetaBadge>{patch.test_kind}</MetaBadge> : null}
          {(quest.frameworks ?? []).map((f) => (
            <MetaBadge key={f}>{f}</MetaBadge>
          ))}
        </div>
        {patch.description ? (
          <p className="mt-4 whitespace-pre-wrap text-sm leading-6 text-quest-ink">
            {patch.description.trim()}
          </p>
        ) : null}
        <div className="mt-4 flex flex-wrap gap-3 text-sm">
          {quest.prereq ? (
            <Link
              href={`/missions/${quest.prereq}`}
              className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent"
            >
              ← 前置：{quest.prereq}
            </Link>
          ) : null}
          {quest.next_lab ? (
            <Link
              href={`/missions/${quest.next_lab}`}
              className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent"
            >
              下一关：{quest.next_lab} →
            </Link>
          ) : null}
        </div>
      </Panel>

      {/* ============================================================ */}
      {/*  ★ Patch task.md (the only required reading)                  */}
      {/* ============================================================ */}
      <Panel title="Patch 任务" eyebrow="task.md / 代码契约">
        {patch.task_md_content ? (
          <pre className="whitespace-pre-wrap font-sans text-sm leading-7 text-quest-ink">
            {patch.task_md_content}
          </pre>
        ) : (
          <p className="text-sm text-quest-muted">
            找不到 {patch.task_md_path}。请确认文件存在。
          </p>
        )}
        <div className="mt-4 flex flex-wrap gap-2 text-sm">
          {patch.starter_file ? (
            <Link
              href={`/source?path=${encodeURIComponent(patch.starter_file)}`}
              className="rounded-full bg-quest-accent px-3 py-1 text-white hover:opacity-90"
            >
              ★ 打开 starter（你要改这个文件）
            </Link>
          ) : null}
          {patch.reference_file ? (
            <Link
              href={`/source?path=${encodeURIComponent(patch.reference_file)}`}
              className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent"
            >
              查看参考解（卡住再开）
            </Link>
          ) : null}
        </div>
      </Panel>

      {/* ============================================================ */}
      {/*  ★ Quiz Gate (must pass before writing patch)                  */}
      {/* ============================================================ */}
      <QuizGate missionId={id} />

      {/* ============================================================ */}
      {/*  ★ Run patch-test (gated visually if quiz not passed)          */}
      {/* ============================================================ */}
      <PatchRunner
        missionId={id}
        initialStatus={patch.status}
        quizPassed={patch.quiz_passed}
      />

      <Panel title="AI 框架理解口试" eyebrow="patch-test 之后">
        <p className="text-sm leading-6 text-quest-ink">
          通过 patch-test 后，把当前 mission id、task.md、source_reading、
          mini_infra_targets、patch 摘要和测试输出发给 AI，让它按五层模型追问；
          如果答不上来，就回到它指出的 MiniInfra/真实源码路径继续读。
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-sm">
          <Link
            href="/prompts"
            className="rounded-full bg-quest-accent px-3 py-1 text-white hover:opacity-90"
          >
            打开 framework tutor 提示卡
          </Link>
          <Link
            href="/source?path=docs%2FAI%E6%A1%86%E6%9E%B6%E7%90%86%E8%A7%A3%E8%AF%84%E4%BC%B0%E6%8C%87%E5%8D%97.md"
            className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent"
          >
            查看评估指南
          </Link>
        </div>
      </Panel>

      {/* ============================================================ */}
      {/*  Resources (links to source/notebooks/mini_infra/tickets)     */}
      {/* ============================================================ */}
      <ResourceSidebar quest={quest} patch={patch} />
    </div>
  );
}

// ----------------------------------------------------------------------------

function ResourceSidebar({
  quest,
  patch,
}: {
  quest: import("@/lib/types").Quest;
  patch: PatchPayload | null;
}) {
  const sourceReading = patch?.source_reading ?? quest.source_reading ?? [];
  const notebooks = patch?.notebooks ?? quest.notebooks ?? [];
  const miniInfra = patch?.mini_infra_targets ?? quest.mini_infra_targets ?? [];
  const tickets = patch?.tickets ?? quest.tickets ?? [];

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Panel title="配套源码研读" eyebrow="可选 · 写完 patch 之后对照真实工程版">
        {sourceReading.length === 0 ? (
          <p className="text-sm text-quest-muted">本关没有配置源码研读路径。</p>
        ) : (
          <ul className="space-y-3 text-sm">
            {sourceReading.map((item, i) => {
              const { path, hitRange } = parseSourceHref(item.repo_path);
              const isRepoFile = Boolean(inferRepoPath(path));
              const href = isRepoFile ? sourceUrl(path, hitRange) : null;
              return (
                <li
                  key={`${item.repo_path}-${i}`}
                  className="rounded-2xl border border-quest-border/70 bg-white/70 p-3"
                >
                  <p className="font-semibold">{item.title}</p>
                  {href ? (
                    <Link href={href} className="mt-1 block font-mono text-xs text-quest-accent hover:underline">
                      {item.repo_path}
                    </Link>
                  ) : (
                    <p className="mt-1 font-mono text-xs text-quest-muted">{item.repo_path}</p>
                  )}
                  {item.focus ? <p className="mt-2 text-quest-muted">{item.focus}</p> : null}
                </li>
              );
            })}
          </ul>
        )}
      </Panel>

      <Panel title="概念 Notebook" eyebrow="可选 · 写 patch 前打基础">
        {notebooks.length === 0 ? (
          <p className="text-sm text-quest-muted">本关没有配置 notebook。</p>
        ) : (
          <ul className="space-y-2 text-sm">
            {notebooks.map((notebook) => (
              <li
                key={notebook}
                className="rounded-2xl border border-quest-border/70 bg-white/70 p-3"
              >
                <div className="font-mono text-xs text-quest-muted">{notebook}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  <Link
                    href={`/notebooks?path=${encodeURIComponent(notebook)}`}
                    className="rounded-full border border-quest-accent px-3 py-1 text-xs text-quest-accent hover:bg-quest-accent hover:text-white"
                  >
                    在控制台静态查看
                  </Link>
                  <Link
                    href={jupyterUrl(notebook)}
                    target="_blank"
                    className="rounded-full border border-quest-border px-3 py-1 text-xs text-quest-ink hover:border-quest-accent"
                  >
                    在 JupyterLab 打开
                  </Link>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel title="MiniInfra 同构骨架" eyebrow="对照 · 写完 patch 看主线版本">
        {miniInfra.length === 0 ? (
          <p className="text-sm text-quest-muted">本关没有配置 mini_infra 目标文件。</p>
        ) : (
          <ul className="space-y-2 text-sm">
            {miniInfra.map((path) => (
              <li
                key={path}
                className="rounded-2xl border border-quest-border/70 bg-white/70 p-3"
              >
                <Link
                  href={`/source?path=${encodeURIComponent(path)}`}
                  className="font-mono text-xs text-quest-accent hover:underline"
                >
                  {path}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel title="Debug Tickets" eyebrow="可选 · 排障训练">
        {tickets.length === 0 ? (
          <p className="text-sm text-quest-muted">本关没有配置 ticket。</p>
        ) : (
          <ul className="space-y-2 text-sm">
            {tickets.map((ticket) => (
              <li key={ticket}>
                <Link
                  href={`/tickets/${ticket}`}
                  className="font-mono text-quest-accent hover:underline"
                >
                  {ticket}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
