import type { ReactNode } from "react";
import Link from "next/link";
import { codeToHtml, type ShikiTransformer } from "shiki";
import { api } from "@/lib/api";
import { sourceUrl, parseLineRange } from "@/lib/source";
import type { LessonPayload, LessonSection as LessonSectionType, LessonSourceRef, SourceReadingItem } from "@/lib/types";
import { Markdown } from "./markdown";
import { SourceReadingCard } from "./source-reading-card";

const SHIKI_LANGUAGE_FALLBACK: Record<string, string> = {
  "cuda-cpp": "cpp",
  tsx: "tsx",
  jsx: "jsx",
  text: "text",
  rst: "text",
  dockerfile: "dockerfile",
  makefile: "makefile",
};

function mapShikiLanguage(lang: string): string {
  return SHIKI_LANGUAGE_FALLBACK[lang] ?? lang;
}

function lineDecorators(start: number, hit: [number, number] | null): ShikiTransformer {
  return {
    name: "lesson-highlight-line",
    line(node, lineNumber) {
      const realLine = start + lineNumber - 1;
      const props = node.properties ?? {};
      props["data-line"] = String(realLine);
      const cls = typeof props.class === "string" ? props.class : "";
      const classes = new Set(cls.split(/\s+/).filter(Boolean));
      classes.add("line");
      if (hit && realLine >= hit[0] && realLine <= hit[1]) {
        classes.add("source-viewer-line-hit");
      }
      props.class = Array.from(classes).join(" ");
      node.properties = props;
    },
  };
}

async function renderSnippet(content: string, language: string, start: number, hit: [number, number] | null) {
  const lang = mapShikiLanguage(language || "text");
  try {
    return await codeToHtml(content, {
      lang,
      theme: "github-light",
      transformers: [lineDecorators(start, hit)],
    });
  } catch {
    return await codeToHtml(content, {
      lang: "text",
      theme: "github-light",
      transformers: [lineDecorators(start, hit)],
    });
  }
}

function repoSourceCheck(path: string): boolean {
  return /^(github_repo|labs|mini_infra|scripts|docs|notebooks|app|simulators|tickets|prompts|quests|data|dashboards)\//.test(path);
}

function normaliseRange(input: [number, number] | string | undefined): [number, number] | null {
  if (
    Array.isArray(input) &&
    input.length === 2 &&
    Number.isFinite(input[0]) &&
    Number.isFinite(input[1])
  ) {
    const a = Math.max(1, Math.floor(input[0]));
    const b = Math.max(1, Math.floor(input[1]));
    return a <= b ? [a, b] : [b, a];
  }
  if (typeof input === "string") return parseLineRange(input);
  return null;
}

function fragmentFor(range: [number, number]): string {
  return range[0] === range[1] ? `${range[0]}` : `${range[0]}-${range[1]}`;
}

async function CodeMoment({ sourceRef }: { sourceRef: LessonSourceRef }) {
  const range = normaliseRange(sourceRef.lines);
  const canRead = repoSourceCheck(sourceRef.repo_path);
  let snippetHtml = "";
  let linkRange: [number, number] = range ?? [1, 1];

  if (range && canRead) {
    try {
      const payload = await api.source(sourceRef.repo_path);
      const lines = (payload.content || "").split("\n");
      const a = Math.max(1, range[0]);
      const b = Math.min(lines.length, range[1]);
      linkRange = [a, b];
      const slice = lines.slice(a - 1, b).join("\n");
      snippetHtml = await renderSnippet(slice, payload.language || "text", a, linkRange);
    } catch {
      snippetHtml = `<pre class="p-3 text-xs text-quest-muted">无法读取 ${sourceRef.repo_path}</pre>`;
    }
  }

  return (
    <div className="grid gap-2 rounded-xl border border-quest-border/60 bg-white/85 p-3 lg:grid-cols-[minmax(0,1.55fr),minmax(0,0.95fr)]">
      <div className="min-w-0">
        <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-quest-muted">
          <span className="rounded-full bg-quest-card-soft px-2 py-1 font-medium text-quest-ink">源码穿插</span>
          {sourceRef.title ? <span className="font-medium text-quest-ink">{sourceRef.title}</span> : null}
          {canRead ? (
            <Link href={range ? sourceUrl(sourceRef.repo_path, linkRange) : sourceUrl(sourceRef.repo_path)} className="font-mono text-quest-accent hover:underline">
              {sourceRef.repo_path}{range ? `#L${fragmentFor(linkRange)}` : ""}
            </Link>
          ) : (
            <span className="font-mono">{sourceRef.repo_path}</span>
          )}
        </div>
        {range && canRead ? (
        <div
          className="source-viewer overflow-auto rounded-lg border border-quest-border/70 bg-[#fdfaf3] text-[11px] leading-5"
            data-source-path={sourceRef.repo_path}
            dangerouslySetInnerHTML={{ __html: snippetHtml }}
          />
        ) : (
          <p className="rounded-xl border border-quest-border/70 bg-[#fdfaf3] p-3 font-mono text-xs text-quest-muted">{sourceRef.repo_path}</p>
        )}
      </div>
      <div className="text-xs leading-5 text-quest-ink lg:border-l lg:border-quest-border/60 lg:pl-3">
        <Markdown compact source={sourceRef.note || "这段代码把刚才的概念落到真实实现里。先看高亮行，不必一口气读完整文件。"} />
      </div>
    </div>
  );
}

function LessonBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-quest-border/60 bg-quest-card-soft/60 p-4">
      <p className="mb-2 text-[11px] uppercase tracking-eyebrow text-quest-muted">{title}</p>
      <div className="text-sm leading-7 text-quest-ink">{children}</div>
    </div>
  );
}

export async function LessonSection({ section, index }: { section: LessonSectionType; index: number }) {
  const sourceRefs = section.source_refs ?? [];
  const explanation = section.plain_explanation ?? section.walkthrough;

  return (
    <article className="rounded-3xl border border-quest-border/70 bg-white/85 p-5 shadow-panel-soft">
      <header>
        <p className="text-[11px] uppercase tracking-eyebrow text-quest-muted">第 {index + 1} 小节</p>
        <h3 className="mt-1 text-lg font-semibold text-quest-ink">{section.title}</h3>
      </header>

      {explanation ? (
        <div className="mt-4 text-sm leading-7 text-quest-ink">
          <Markdown source={explanation} />
        </div>
      ) : null}

      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        {section.mental_model ? (
          <LessonBlock title="直觉模型">
            <Markdown source={section.mental_model} />
          </LessonBlock>
        ) : null}
        {section.why_it_matters ? (
          <LessonBlock title="为什么重要">
            <Markdown source={section.why_it_matters} />
          </LessonBlock>
        ) : null}
      </div>

      {sourceRefs.length > 0 ? (
        <div className="mt-4 space-y-3">
          {sourceRefs.map((sourceRef, i) => (
            <CodeMoment key={`${sourceRef.repo_path}-${i}`} sourceRef={sourceRef} />
          ))}
        </div>
      ) : null}

      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        {section.common_confusions && section.common_confusions.length > 0 ? (
          <LessonBlock title="常见误解">
            <ul className="list-disc space-y-1 pl-5">
              {section.common_confusions.map((item, i) => <li key={i}>{item}</li>)}
            </ul>
          </LessonBlock>
        ) : null}
        {section.checkpoint_questions && section.checkpoint_questions.length > 0 ? (
          <LessonBlock title="读完自问">
            <ul className="list-disc space-y-1 pl-5">
              {section.checkpoint_questions.map((item, i) => <li key={i}>{item}</li>)}
            </ul>
          </LessonBlock>
        ) : null}
      </div>
    </article>
  );
}

export async function LessonRenderer({ lesson, fallbackSourceReading = [] }: { lesson?: LessonPayload; fallbackSourceReading?: SourceReadingItem[] }) {
  const sections = lesson?.sections ?? [];

  if (!lesson && fallbackSourceReading.length > 0) {
    return (
      <div className="space-y-4">
        {fallbackSourceReading.map((item) => (
          <SourceReadingCard key={item.repo_path} item={item} />
        ))}
      </div>
    );
  }

  if (!lesson) {
    return <p className="text-sm text-quest-muted">本关讲义还在补写中。</p>;
  }

  return (
    <div className="space-y-4">
      {lesson.opening ? (
        <div className="rounded-3xl border border-quest-border/70 bg-quest-card-soft p-5 text-sm leading-7 text-quest-ink">
          <p className="mb-2 text-[11px] uppercase tracking-eyebrow text-quest-muted">开场白</p>
          <Markdown source={lesson.opening} />
        </div>
      ) : null}
      {sections.map((section, index) => (
        <LessonSection key={`${section.title}-${index}`} section={section} index={index} />
      ))}
    </div>
  );
}
