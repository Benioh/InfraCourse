import Link from "next/link";
import { api } from "@/lib/api";
import { Panel } from "@/components/ui";
import CodeEditor from "@/components/code-editor";
import { Markdown } from "@/components/markdown";
import QuizGate from "@/components/quiz-gate";
import RegisterPageContext from "@/components/register-page-context";
import { SourceReadingCard } from "@/components/source-reading-card";
import { sourceUrl } from "@/lib/source";
import { withPublicPrefix } from "@/lib/runtime";
import type { PatchPayload, StarterPayload } from "@/lib/types";

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

function StepHeader({
  step,
  title,
  hint,
}: {
  step: string;
  title: string;
  hint: string;
}) {
  return (
    <header className="mb-3 flex items-baseline gap-3">
      <span className="rounded-full bg-quest-accent px-2.5 py-0.5 text-xs font-semibold text-white">
        {step}
      </span>
      <h2 className="text-lg font-semibold text-quest-ink">{title}</h2>
      <span className="text-xs text-quest-muted">{hint}</span>
    </header>
  );
}

export default async function MissionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  const [quest, patch] = await Promise.all([
    api.quest(id),
    api.patch(id).catch<PatchPayload | null>(() => null),
  ]);

  if (!patch) {
    return (
      <div className="space-y-4">
        <RegisterPageContext ctx={{ page_kind: "mission", mission_id: id }} />
        <Panel title={`${quest.level ?? ""} · ${quest.title}`} eyebrow={quest.role}>
          <p className="text-sm text-quest-muted">
            本关尚未配置 Patch Track（quests/{id}.yaml 缺少 <code>patch:</code> 块）。
            目前只能阅读源码与 notebook，作者完成 patch 设计后这里会出现新的任务面板。
          </p>
        </Panel>
      </div>
    );
  }

  const starter = await api
    .starter(id)
    .catch<StarterPayload | null>(() => null);

  const sourceReading = patch.source_reading ?? quest.source_reading ?? [];
  const notebooks = patch.notebooks ?? quest.notebooks ?? [];
  const miniInfra = patch.mini_infra_targets ?? quest.mini_infra_targets ?? [];
  const tickets = patch.tickets ?? quest.tickets ?? [];

  const summaryForTutor = [
    `${quest.level ?? ""} ${quest.title}`,
    patch.description ? `任务：${patch.description.trim().slice(0, 200)}` : "",
    sourceReading.length ? `源码研读：${sourceReading.map((s) => s.repo_path).join(", ")}` : "",
    starter?.path ? `Starter：${starter.path}` : "",
  ]
    .filter(Boolean)
    .join("\n");

  return (
    <div className="space-y-6">
      <RegisterPageContext
        ctx={{
          page_kind: "mission",
          mission_id: id,
          page_summary: summaryForTutor,
        }}
      />

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
        <div className="mt-4 rounded-2xl border border-quest-border bg-quest-card-soft p-4 text-sm leading-7 text-quest-ink">
          <p className="text-[11px] uppercase tracking-eyebrow text-quest-muted">本关学习路径</p>
          <ol className="mt-2 list-decimal space-y-1 pl-5">
            <li>跟着 <strong>步骤 1</strong> 读源码 / 关键行讲解</li>
            <li>选做 <strong>步骤 2</strong> 概念 notebook</li>
            <li>过 <strong>步骤 3</strong> Quiz Gate（必须通过）</li>
            <li>在 <strong>步骤 4</strong> 直接改 starter，写完跑 patch-test</li>
          </ol>
        </div>
        <div className="mt-4 flex flex-wrap gap-3 text-sm">
          {quest.prereq ? (
            <Link
              href={withPublicPrefix(`/missions/${quest.prereq}`)}
              className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent"
            >
              ← 前置：{quest.prereq}
            </Link>
          ) : null}
          {quest.next_lab ? (
            <Link
              href={withPublicPrefix(`/missions/${quest.next_lab}`)}
              className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent"
            >
              下一关：{quest.next_lab} →
            </Link>
          ) : null}
        </div>
      </Panel>

      {/* ============================================================ */}
      {/*  Step 1 — Source reading (annotated)                          */}
      {/* ============================================================ */}
      <Panel title="① 源码研读" eyebrow="跟着导师讲解逐段读懂关键代码">
        <StepHeader
          step="STEP 1"
          title="先读详细讲解，再点高亮跳到完整文件对照"
          hint="📖 讲解部分写给小白；右侧 note 是这段代码的工程含义"
        />
        {sourceReading.length === 0 ? (
          <p className="text-sm text-quest-muted">
            本关没有配置源码研读路径。先看下面的任务说明，再决定要不要去 mini_infra 找对位实现。
          </p>
        ) : (
          <div className="space-y-4">
            {sourceReading.map((item, i) => (
              /* @ts-expect-error async server component */
              <SourceReadingCard key={`${item.repo_path}-${i}`} item={item} />
            ))}
          </div>
        )}
      </Panel>

      {/* ============================================================ */}
      {/*  Step 2 — Notebook (optional)                                 */}
      {/* ============================================================ */}
      <Panel title="② 概念 Notebook" eyebrow="可选 · 写 patch 前打基础">
        <StepHeader
          step="STEP 2"
          title="跑 notebook 把数据/张量摸一遍"
          hint="可选 — 已经懂的可以直接跳到下一步"
        />
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
                    href={withPublicPrefix(`/notebooks?path=${encodeURIComponent(notebook)}`)}
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

      {/* ============================================================ */}
      {/*  Step 3 — Quiz Gate                                           */}
      {/* ============================================================ */}
      <Panel title="③ 概念关 Quiz Gate" eyebrow="必须通过才进入写代码">
        <StepHeader
          step="STEP 3"
          title="先过概念关再写代码"
          hint="patch 通过 ≠ 你懂；这一步是必经路口"
        />
        <QuizGate missionId={id} />
      </Panel>

      {/* ============================================================ */}
      {/*  Step 4 — Task + editable starter                              */}
      {/* ============================================================ */}
      <Panel title="④ 写代码 · Patch" eyebrow="task.md 契约 + 直接改 starter">
        <StepHeader
          step="STEP 4"
          title="按契约写代码，跑测试"
          hint="可以打开右侧 AI 助手切换到「写代码」模式陪你 vibe"
        />

        <div className="mb-5 rounded-3xl border border-quest-border/70 bg-quest-card-soft p-5">
          <div className="flex items-baseline justify-between gap-3">
            <h3 className="text-base font-semibold text-quest-ink">任务契约（task.md）</h3>
            <span className="font-mono text-[11px] text-quest-muted">{patch.task_md_path}</span>
          </div>
          {patch.task_md_content ? (
            <Markdown source={patch.task_md_content} className="mt-2" />
          ) : (
            <p className="mt-2 text-sm text-quest-muted">
              找不到 {patch.task_md_path}。请确认文件存在。
            </p>
          )}
          <div className="mt-4 flex flex-wrap gap-2 text-sm">
            {patch.reference_file ? (
              <Link
                href={withPublicPrefix(`/source?path=${encodeURIComponent(patch.reference_file)}`)}
                className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent"
              >
                查看参考解（卡住再开）
              </Link>
            ) : null}
          </div>
        </div>

        {starter && patch.starter_file ? (
          <CodeEditor
            missionId={id}
            initialContent={starter.content}
            path={patch.starter_file}
            language={starter.language}
            quizPassed={patch.quiz_passed}
            initialPatchStatus={patch.status ?? null}
          />
        ) : (
          <p className="text-sm text-quest-muted">
            本关没有配置 starter_file，无法在线编辑。
          </p>
        )}
      </Panel>

      {/* ============================================================ */}
      {/*  AI 框架理解口试                                                */}
      {/* ============================================================ */}
      <Panel title="⑤ AI 框架理解口试" eyebrow="patch-test 之后">
        <p className="text-sm leading-6 text-quest-ink">
          通过 patch-test 后，把当前 mission id、task.md、source_reading、
          mini_infra_targets、patch 摘要和测试输出发给 AI，让它按五层模型追问；
          如果答不上来，就回到它指出的 MiniInfra/真实源码路径继续读。
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-sm">
          <Link
            href={withPublicPrefix("/prompts")}
            className="rounded-full bg-quest-accent px-3 py-1 text-white hover:opacity-90"
          >
            打开 framework tutor 提示卡
          </Link>
          <Link
            href={withPublicPrefix("/source?path=docs%2FAI_tutor%2FAI%E6%A1%86%E6%9E%B6%E7%90%86%E8%A7%A3%E8%AF%84%E4%BC%B0%E6%8C%87%E5%8D%97.md")}
            className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent"
          >
            查看评估指南
          </Link>
        </div>
      </Panel>

      {/* ============================================================ */}
      {/*  Resources (secondary)                                         */}
      {/* ============================================================ */}
      <div className="grid gap-6 lg:grid-cols-2">
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
                    href={withPublicPrefix(`/source?path=${encodeURIComponent(path)}`)}
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
                    href={withPublicPrefix(`/tickets/${ticket}`)}
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
    </div>
  );
}
