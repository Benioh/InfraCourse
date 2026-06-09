import Link from "next/link";
import { api } from "@/lib/api";
import { Panel, CommandBlock } from "@/components/ui";
import CodeEditor from "@/components/code-editor";
import { LessonRenderer } from "@/components/lesson-section";
import { Markdown } from "@/components/markdown";
import QuizGate from "@/components/quiz-gate";
import RegisterPageContext from "@/components/register-page-context";
import { withPublicPrefix } from "@/lib/runtime";
import type { LessonDocItem, PatchPayload, Quest, StarterPayload } from "@/lib/types";

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

function StepHeader({ step, title, hint }: { step: string; title: string; hint: string }) {
  return (
    <header className="mb-3 flex flex-col gap-2 lg:flex-row lg:items-baseline lg:gap-3">
      <span className="w-fit rounded-full bg-quest-accent px-2.5 py-0.5 text-xs font-semibold text-white">
        {step}
      </span>
      <h2 className="text-lg font-semibold text-quest-ink">{title}</h2>
      <span className="text-xs text-quest-muted">{hint}</span>
    </header>
  );
}

function nextAction(patch: PatchPayload | null): string {
  if (!patch) return "先读本关讲义，按页面里的命令完成环境验证。";
  if (!patch.quiz_passed) return "先读本关讲义，再完成 Quiz。";
  if (!patch.status?.passed) return "现在可以写 Patch，并运行最后验证命令。";
  return "做 AI 口试，能讲清楚后进入下一关。";
}

function CommandsPanel({ commands }: { commands?: Record<string, string> }) {
  const entries = Object.entries(commands ?? {});
  if (!entries.length) return null;
  return (
    <Panel title="本关命令" eyebrow="最后验证 / 不要一上来就跑">
      <p className="mb-4 text-sm leading-6 text-quest-muted">
        命令是用来验证你读完讲义后的判断，不是第一步。先知道每条命令在检查什么，再运行它。
      </p>
      <div className="space-y-3">
        {entries.map(([name, command]) => (
          <div key={name} className="rounded-2xl border border-quest-border/70 bg-white/70 p-3">
            <p className="mb-2 text-xs uppercase tracking-eyebrow text-quest-muted">{name}</p>
            <CommandBlock command={command} />
          </div>
        ))}
      </div>
    </Panel>
  );
}

async function LessonDocs({ docs }: { docs: LessonDocItem[] }) {
  if (!docs.length) return null;
  const loaded = await Promise.all(
    docs.map(async (doc) => {
      try {
        const payload = await api.source(doc.path);
        return { ...doc, content: payload.content ?? "" };
      } catch {
        return { ...doc, content: "" };
      }
    }),
  );
  return (
    <div className="mb-5 space-y-4">
      {loaded.map((doc) => (
        <article key={doc.path} className="rounded-2xl border border-quest-border/70 bg-white/85 p-5">
          <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
            <div>
              <h3 className="text-base font-semibold text-quest-ink">{doc.title}</h3>
              {doc.description ? <p className="mt-1 text-sm text-quest-muted">{doc.description}</p> : null}
            </div>
            <span className="font-mono text-[11px] text-quest-muted">{doc.path}</span>
          </div>
          {doc.content ? (
            <Markdown source={doc.content} basePath={doc.path} />
          ) : (
            <p className="text-sm text-quest-muted">未能读取讲义文档：{doc.path}</p>
          )}
        </article>
      ))}
    </div>
  );
}

function ProjectPanel({ quest }: { quest: Quest }) {
  const project = quest.project;
  if (!project && !(quest.learning_goals?.length || quest.success_criteria?.length)) return null;
  return (
    <Panel title="这关最后要交付什么" eyebrow="Project / 读完讲义后的目标">
      {project?.scenario ? <p className="text-sm leading-7 text-quest-ink">{project.scenario}</p> : null}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        {quest.learning_goals?.length ? (
          <div className="rounded-2xl border border-quest-border/70 bg-white/70 p-4">
            <p className="text-[11px] uppercase tracking-eyebrow text-quest-muted">你会学会</p>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-quest-ink">
              {quest.learning_goals.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </div>
        ) : null}
        {project?.deliverables?.length ? (
          <div className="rounded-2xl border border-quest-border/70 bg-white/70 p-4">
            <p className="text-[11px] uppercase tracking-eyebrow text-quest-muted">交付物</p>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-quest-ink">
              {project.deliverables.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </div>
        ) : null}
        {quest.success_criteria?.length ? (
          <div className="rounded-2xl border border-quest-border/70 bg-white/70 p-4 lg:col-span-2">
            <p className="text-[11px] uppercase tracking-eyebrow text-quest-muted">完成标准</p>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-quest-ink">
              {quest.success_criteria.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </div>
        ) : null}
      </div>
    </Panel>
  );
}

export default async function MissionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  const [quest, patch] = await Promise.all([
    api.quest(id),
    api.patch(id).catch<PatchPayload | null>(() => null),
  ]);

  const starter = patch
    ? await api.starter(id).catch<StarterPayload | null>(() => null)
    : null;

  const lesson = patch?.lesson ?? quest.lesson;
  const lessonDocs = patch?.lesson_docs ?? quest.lesson_docs ?? [];
  const sourceReading = patch?.source_reading ?? quest.source_reading ?? [];
  const notebooks = patch?.notebooks ?? quest.notebooks ?? [];
  const miniInfra = patch?.mini_infra_targets ?? quest.mini_infra_targets ?? [];
  const tickets = patch?.tickets ?? quest.tickets ?? [];

  const summaryForTutor = [
    `${quest.level ?? ""} ${quest.title}`,
    lesson?.opening ? `讲义：${lesson.opening.trim().slice(0, 200)}` : "",
    patch?.description ? `Lab：${patch.description.trim().slice(0, 200)}` : "",
    sourceReading.length ? `源码穿插：${sourceReading.map((s) => s.repo_path).join(", ")}` : "",
    starter?.path ? `Starter：${starter.path}` : "",
  ].filter(Boolean).join("\n");

  return (
    <div className="space-y-6">
      <RegisterPageContext ctx={{ page_kind: "mission", mission_id: id, page_summary: summaryForTutor }} />

      <Panel title={`${quest.level ?? ""} · ${quest.title}`} eyebrow={quest.role ?? "Mission"}>
        <div className="flex flex-wrap items-center gap-2 text-sm text-quest-muted">
          <MetaBadge>章节：{quest.act ?? "—"}</MetaBadge>
          <MetaBadge>预计 {quest.estimated_minutes ?? "?"} 分钟</MetaBadge>
          {quest.no_gpu_friendly ? <MetaBadge>0 GPU 可完成</MetaBadge> : null}
          {patch?.test_count ? <MetaBadge>{patch.test_count} 个测试</MetaBadge> : null}
          {patch?.test_kind ? <MetaBadge>{patch.test_kind}</MetaBadge> : null}
          {(quest.frameworks ?? []).map((f) => <MetaBadge key={f}>{f}</MetaBadge>)}
        </div>
        <div className="mt-4 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-2xl border border-quest-border bg-quest-card-soft p-4 text-sm leading-7 text-quest-ink">
            <p className="text-[11px] uppercase tracking-eyebrow text-quest-muted">本关怎么玩</p>
            <ol className="mt-2 list-decimal space-y-1 pl-5">
              <li>第一遍不要急着写代码。先把 <strong>本关讲义</strong> 读完。</li>
              <li>看到源码片段时，只看高亮行和旁边解释；完整文件是回看用的。</li>
              <li>Notebook 用来验证讲义直觉，可以晚一点再做。</li>
              <li>Quiz 和 Lab 都来自讲义，不会考没有讲过的源码细节。</li>
            </ol>
          </div>
          <div className="rounded-2xl border border-quest-accent/40 bg-white p-4 text-sm leading-7 text-quest-ink">
            <p className="text-[11px] uppercase tracking-eyebrow text-quest-muted">当前推荐动作</p>
            <p className="mt-2 text-base font-semibold">{nextAction(patch)}</p>
            <div className="mt-4 flex flex-wrap gap-2">
              <Link href="#lesson" className="rounded-full bg-quest-accent px-3 py-1 text-xs text-white hover:opacity-90">进入本关讲义</Link>
              {patch ? <Link href="#quiz" className="rounded-full border border-quest-border px-3 py-1 text-xs text-quest-muted hover:border-quest-accent">跳到 Quiz</Link> : null}
              {patch ? <Link href="#patch" className="rounded-full border border-quest-border px-3 py-1 text-xs text-quest-muted hover:border-quest-accent">跳到 Lab</Link> : null}
            </div>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-3 text-sm">
          {quest.prereq ? <Link href={withPublicPrefix(`/missions/${quest.prereq}`)} className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent">← 前置：{quest.prereq}</Link> : null}
          {quest.next_lab ? <Link href={withPublicPrefix(`/missions/${quest.next_lab}`)} className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent">下一关：{quest.next_lab} →</Link> : null}
        </div>
      </Panel>

      <section id="lesson">
        <Panel title="① 本关讲义" eyebrow="从零开始讲清楚：它是什么、为什么需要、真实系统怎么做">
          <StepHeader step="STEP 1" title="先像上课一样读懂背景，再看源码片段怎么落地" hint="讲义是主线；源码只是讲到对应概念时的证据" />
          <LessonDocs docs={lessonDocs} />
          <LessonRenderer lesson={lesson} fallbackSourceReading={sourceReading} />
        </Panel>
      </section>

      <section id="notebook">
        <Panel title="② 小实验 / Notebook" eyebrow="可选 · 用来验证讲义里的直觉">
          <StepHeader step="STEP 2" title="把讲义里的概念跑一遍" hint="已经懂的可以先跳到 Quiz，卡住再回来做实验" />
          {notebooks.length === 0 ? <p className="text-sm text-quest-muted">本关没有配置 notebook。</p> : (
            <ul className="space-y-2 text-sm">
              {notebooks.map((notebook) => (
                <li key={notebook} className="rounded-2xl border border-quest-border/70 bg-white/70 p-3">
                  <div className="font-mono text-xs text-quest-muted">{notebook}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Link href={withPublicPrefix(`/notebooks?path=${encodeURIComponent(notebook)}`)} className="rounded-full border border-quest-accent px-3 py-1 text-xs text-quest-accent hover:bg-quest-accent hover:text-white">在控制台静态查看</Link>
                    <Link href={jupyterUrl(notebook)} target="_blank" className="rounded-full border border-quest-border px-3 py-1 text-xs text-quest-ink hover:border-quest-accent">在 JupyterLab 打开</Link>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </section>

      {patch ? (
        <>
          <section id="quiz">
            <Panel title="③ 概念关 Quiz" eyebrow="只检查讲义里的关键概念">
              <StepHeader step="STEP 3" title="先确认自己真的懂，再写代码" hint="这些题不会考没讲过的源码细节" />
              <p className="mb-4 rounded-2xl border border-quest-border/70 bg-quest-card-soft p-3 text-sm leading-6 text-quest-muted">这些题只检查讲义里的关键概念。不会考没讲过的源码细节。</p>
              <QuizGate missionId={id} />
            </Panel>
          </section>

          <section id="patch">
            <Panel title="④ 写代码 · Lab" eyebrow="把讲义里的核心机制缩小成可测试实现">
              <StepHeader step="STEP 4" title="按契约写一个小补丁，跑最后验证命令" hint="你不是在复刻整个框架，只是在实现讲义里最核心的一小块" />
              <p className="mb-4 rounded-2xl border border-quest-border/70 bg-quest-card-soft p-3 text-sm leading-6 text-quest-muted">这个 Lab 只实现讲义里最核心的一小块机制。你不是在复刻整个框架。</p>

              <div className="mb-5 rounded-3xl border border-quest-border/70 bg-quest-card-soft p-5">
                <div className="flex items-baseline justify-between gap-3">
                  <h3 className="text-base font-semibold text-quest-ink">任务契约（task.md）</h3>
                  <span className="font-mono text-[11px] text-quest-muted">{patch.task_md_path}</span>
                </div>
                {patch.task_md_content ? <Markdown source={patch.task_md_content} className="mt-2" /> : <p className="mt-2 text-sm text-quest-muted">找不到 {patch.task_md_path}。请确认文件存在。</p>}
                <div className="mt-4 flex flex-wrap gap-2 text-sm">
                  {patch.reference_file ? <Link href={withPublicPrefix(`/source?path=${encodeURIComponent(patch.reference_file)}`)} className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent">查看参考解（卡住再开）</Link> : null}
                </div>
              </div>

              {starter && patch.starter_file ? (
                <CodeEditor missionId={id} initialContent={starter.content} path={patch.starter_file} language={starter.language} quizPassed={patch.quiz_passed} initialPatchStatus={patch.status ?? null} />
              ) : <p className="text-sm text-quest-muted">本关没有配置 starter_file，无法在线编辑。</p>}
            </Panel>
          </section>

          <section id="oral-check">
            <Panel title="⑤ AI 框架理解口试" eyebrow="patch-test 之后">
              <p className="text-sm leading-6 text-quest-ink">通过 patch-test 后，把当前 mission id、讲义、task.md、源码穿插、patch 摘要和测试输出发给 AI，让它按五层模型追问；如果答不上来，就回到它指出的讲义小节或源码路径继续读。</p>
              <div className="mt-4 flex flex-wrap gap-2 text-sm">
                <Link href={withPublicPrefix("/prompts")} className="rounded-full bg-quest-accent px-3 py-1 text-white hover:opacity-90">打开 framework tutor 提示卡</Link>
                <Link href={withPublicPrefix("/source?path=docs%2FAI_tutor%2FAI%E6%A1%86%E6%9E%B6%E7%90%86%E8%A7%A3%E8%AF%84%E4%BC%B0%E6%8C%87%E5%8D%97.md")} className="rounded-full border border-quest-border px-3 py-1 text-quest-muted hover:border-quest-accent hover:text-quest-accent">查看评估指南</Link>
              </div>
            </Panel>
          </section>
        </>
      ) : (
        <>
          <ProjectPanel quest={quest} />
          <CommandsPanel commands={quest.commands} />
        </>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="MiniInfra 同构骨架" eyebrow="对照 · 读完讲义再看主线版本">
          {miniInfra.length === 0 ? <p className="text-sm text-quest-muted">本关没有配置 mini_infra 目标文件。</p> : (
            <ul className="space-y-2 text-sm">
              {miniInfra.map((path) => (
                <li key={path} className="rounded-2xl border border-quest-border/70 bg-white/70 p-3">
                  <Link href={withPublicPrefix(`/source?path=${encodeURIComponent(path)}`)} className="font-mono text-xs text-quest-accent hover:underline">{path}</Link>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Debug Tickets" eyebrow="可选 · 排障训练">
          {tickets.length === 0 ? <p className="text-sm text-quest-muted">本关没有配置 ticket。</p> : (
            <ul className="space-y-2 text-sm">
              {tickets.map((ticket) => (
                <li key={ticket}><Link href={withPublicPrefix(`/tickets/${ticket}`)} className="font-mono text-quest-accent hover:underline">{ticket}</Link></li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
