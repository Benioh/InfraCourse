import Link from "next/link";
import { codeToHtml, type ShikiTransformer } from "shiki";
import { api } from "@/lib/api";
import { sourceUrl, parseLineRange } from "@/lib/source";
import { Markdown } from "./markdown";
import type { SourceReadingItem } from "@/lib/types";

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
    name: "highlight-line",
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

async function renderSnippet(
  content: string,
  language: string,
  start: number,
  hit: [number, number] | null,
): Promise<string> {
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
  // Same heuristic the rest of the UI uses: only paths under known roots.
  return /^(github_repo|labs|mini_infra|scripts|docs|notebooks|app|simulators|tickets|prompts|quests|data|dashboards)\//.test(
    path,
  );
}

function fragmentFor(range: [number, number]): string {
  return range[0] === range[1] ? `${range[0]}` : `${range[0]}-${range[1]}`;
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
  if (typeof input === "string") {
    return parseLineRange(input);
  }
  return null;
}

async function HighlightBlock({
  path,
  range,
  note,
  title,
}: {
  path: string;
  range: [number, number];
  note: string;
  title?: string;
}) {
  let snippetHtml = "";
  let language = "text";
  try {
    const payload = await api.source(path);
    language = payload.language || "text";
    const lines = (payload.content || "").split("\n");
    const a = Math.max(1, range[0]);
    const b = Math.min(lines.length, range[1]);
    const slice = lines.slice(a - 1, b).join("\n");
    snippetHtml = await renderSnippet(slice, language, a, range);
  } catch {
    snippetHtml = `<pre class="p-3 text-xs text-quest-muted">无法读取 ${path}</pre>`;
  }

  return (
    <div className="grid gap-3 rounded-2xl border border-quest-border/60 bg-white/70 p-4 lg:grid-cols-[minmax(0,1.6fr),minmax(0,1fr)]">
      <div className="min-w-0">
        <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-quest-muted">
          {title ? <span className="font-medium text-quest-ink">{title}</span> : null}
          <Link
            href={sourceUrl(path, range)}
            className="font-mono text-quest-accent hover:underline"
          >
            {path}#L{fragmentFor(range)}
          </Link>
        </div>
        <div
          className="source-viewer overflow-auto rounded-xl border border-quest-border/70 bg-[#fdfaf3] text-[12px] leading-6"
          data-source-path={path}
          dangerouslySetInnerHTML={{ __html: snippetHtml }}
        />
      </div>
      <div className="text-sm leading-7 text-quest-ink lg:pl-3 lg:border-l lg:border-quest-border/60">
        <Markdown source={note} />
      </div>
    </div>
  );
}

export async function SourceReadingCard({ item }: { item: SourceReadingItem }) {
  const isRepoFile = repoSourceCheck(item.repo_path);
  const linkHref = isRepoFile ? sourceUrl(item.repo_path) : null;
  const highlights = item.highlights ?? [];

  return (
    <article className="rounded-3xl border border-quest-border/70 bg-white/80 p-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h4 className="text-base font-semibold text-quest-ink">{item.title}</h4>
          {linkHref ? (
            <Link href={linkHref} className="font-mono text-xs text-quest-accent hover:underline">
              {item.repo_path}
            </Link>
          ) : (
            <p className="font-mono text-xs text-quest-muted">{item.repo_path}</p>
          )}
        </div>
        {linkHref ? (
          <Link
            href={linkHref}
            className="rounded-full border border-quest-border px-3 py-1 text-xs text-quest-ink hover:border-quest-accent"
          >
            查看完整文件 →
          </Link>
        ) : null}
      </header>

      {item.focus ? (
        <p className="mt-3 rounded-2xl bg-quest-card-soft px-3 py-2 text-sm text-quest-muted">
          🎯 {item.focus}
        </p>
      ) : null}

      {item.walkthrough ? (
        <div className="mt-4 rounded-2xl border border-quest-border/60 bg-quest-card-soft/60 p-4">
          <p className="mb-2 text-[11px] uppercase tracking-eyebrow text-quest-muted">
            📖 详细讲解
          </p>
          <Markdown source={item.walkthrough} />
        </div>
      ) : null}

      {highlights.length > 0 ? (
        <div className="mt-4 space-y-3">
          {highlights.map((h, i) => {
            const range = normaliseRange(h.lines);
            if (!range || !isRepoFile) return null;
            return (
              /* @ts-expect-error async server component in a list */
              <HighlightBlock
                key={`${item.repo_path}-${i}`}
                path={item.repo_path}
                range={range}
                note={h.note}
                title={h.title}
              />
            );
          })}
        </div>
      ) : (
        <p className="mt-4 text-sm text-quest-muted">
          这个文件作者还没标关键行，先按 Focus 自己读，把不懂的圈起来问 AI Tutor。
        </p>
      )}

      {item.questions && item.questions.length > 0 ? (
        <div className="mt-4 rounded-2xl border border-quest-border/60 bg-quest-card-soft p-3">
          <p className="text-[11px] uppercase tracking-eyebrow text-quest-muted">读完自问</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-quest-ink">
            {item.questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </article>
  );
}
