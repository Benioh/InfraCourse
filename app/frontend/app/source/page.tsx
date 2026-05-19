import Link from "next/link";
import { notFound } from "next/navigation";
import { codeToHtml, type ShikiTransformer } from "shiki";
import { api } from "@/lib/api";
import { Panel } from "@/components/ui";
import { CopyButton } from "@/components/copy-button";
import RegisterPageContext from "@/components/register-page-context";
import { buildBreadcrumbs, parseLineRange, sourceUrl } from "@/lib/source";
import { withPublicPrefix } from "@/lib/runtime";
import type { SourcePayload, SourceTreeEntry, SourceTreePayload } from "@/lib/types";

type SearchParams = Promise<{ path?: string; dir?: string; lines?: string }>;

const SHIKI_LANGUAGE_FALLBACK: Record<string, string> = {
  "cuda-cpp": "cpp",
  "tsx": "tsx",
  "jsx": "jsx",
  "objc": "objective-c",
  "text": "text",
  "rst": "text",
  "dockerfile": "dockerfile",
  "makefile": "makefile",
};

function mapShikiLanguage(lang: string): string {
  return SHIKI_LANGUAGE_FALLBACK[lang] ?? lang;
}

function lineDecorators(hit: [number, number] | null): ShikiTransformer {
  return {
    name: "quest-line-decorators",
    line(node, lineNumber) {
      const props = node.properties ?? {};
      props["data-line"] = String(lineNumber);
      const existing = typeof props.class === "string" ? props.class : "";
      const classes = new Set(existing.split(/\s+/).filter(Boolean));
      classes.add("line");
      if (hit && lineNumber >= hit[0] && lineNumber <= hit[1]) {
        classes.add("source-viewer-line-hit");
      }
      props.class = Array.from(classes).join(" ");
      node.properties = props;
    },
  };
}

async function renderCode(payload: SourcePayload, hit: [number, number] | null): Promise<string> {
  const lang = mapShikiLanguage(payload.language || "text");
  try {
    return await codeToHtml(payload.content, {
      lang,
      theme: "github-light",
      transformers: [lineDecorators(hit)],
    });
  } catch {
    return await codeToHtml(payload.content, {
      lang: "text",
      theme: "github-light",
      transformers: [lineDecorators(hit)],
    });
  }
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function Breadcrumbs({ path, isDir }: { path: string; isDir: boolean }) {
  const crumbs = buildBreadcrumbs(path);
  return (
    <div className="flex flex-wrap items-center gap-1 font-mono text-xs text-quest-muted">
      <Link href={withPublicPrefix("/source")} className="text-quest-accent hover:underline">
        仓库根
      </Link>
      {crumbs.map((crumb, idx) => {
        const showAsFile = crumb.isLast && !isDir;
        const href = showAsFile
          ? sourceUrl(crumbPath(crumbs, idx))
          : withPublicPrefix(`/source?dir=${encodeURIComponent(crumbPath(crumbs, idx))}`);
        return (
          <span key={`${crumb.name}-${idx}`} className="flex items-center gap-1">
            <span className="text-quest-muted">/</span>
            <Link href={href} className="text-quest-accent hover:underline">
              {crumb.name}
            </Link>
          </span>
        );
      })}
    </div>
  );
}

function crumbPath(crumbs: { name: string }[], upTo: number): string {
  return crumbs.slice(0, upTo + 1).map((c) => c.name).join("/");
}

function TreeView({ tree }: { tree: SourceTreePayload }) {
  return (
    <ul className="divide-y divide-quest-border/60 overflow-hidden rounded-2xl border border-quest-border/70 bg-white/80">
      {tree.entries.length === 0 ? (
        <li className="p-4 text-sm text-quest-muted">目录为空。</li>
      ) : null}
      {tree.entries.map((entry: SourceTreeEntry) => {
        const href =
          entry.type === "dir"
            ? withPublicPrefix(`/source?dir=${encodeURIComponent(entry.path)}`)
            : entry.type === "file"
              ? sourceUrl(entry.path)
              : null;
        const icon = entry.type === "dir" ? "📁" : entry.type === "binary" ? "⧫" : "📄";
        const sizeLabel = entry.type === "file" && entry.size != null ? formatBytes(entry.size) : "";
        return (
          <li key={entry.path} className="flex items-center justify-between px-4 py-2 text-sm">
            <span className="flex items-center gap-2 font-mono">
              <span>{icon}</span>
              {href ? (
                <Link href={href} className="text-quest-accent hover:underline">
                  {entry.name}
                </Link>
              ) : (
                <span className="text-quest-muted">{entry.name}</span>
              )}
            </span>
            <span className="text-xs text-quest-muted">
              {entry.type === "binary" ? "二进制" : sizeLabel}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

async function FileView({ path, hit }: { path: string; hit: [number, number] | null }) {
  let payload: SourcePayload;
  try {
    payload = await api.source(path);
  } catch {
    notFound();
  }

  if (payload.truncated) {
    return (
      <Panel title={payload.path} eyebrow="Source / 源码查看">
        <Breadcrumbs path={payload.path} isDir={false} />
        <p className="mt-4 text-sm text-quest-warn">
          文件过大（{formatBytes(payload.size)}），未在浏览器内渲染。{payload.reason ?? ""}
        </p>
      </Panel>
    );
  }

  const html = await renderCode(payload, hit);
  const dirPath = payload.path.includes("/") ? payload.path.slice(0, payload.path.lastIndexOf("/")) : "";

  return (
    <div data-ai-context-root>
      <RegisterPageContext
        ctx={{
          page_kind: "source",
          source_path: payload.path,
          lines: hit ?? undefined,
          page_summary: `source=${payload.path}${
            hit ? `#L${hit[0]}-${hit[1]}` : ""
          } · ${payload.language} · ${payload.line_count} lines`,
        }}
      />
      <Panel title={payload.path} eyebrow="Source / 源码查看">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <Breadcrumbs path={payload.path} isDir={false} />
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-quest-border px-3 py-1 text-xs text-quest-muted">
              {payload.language} · {payload.line_count} 行 · {formatBytes(payload.size)}
            </span>
            {dirPath ? (
              <Link
                href={withPublicPrefix(`/source?dir=${encodeURIComponent(dirPath)}`)}
                className="rounded-full border border-quest-border px-3 py-1 text-xs text-quest-ink hover:border-quest-accent"
              >
                同目录文件
              </Link>
            ) : null}
            <CopyButton text={payload.content} label="复制全文" />
          </div>
        </div>
        {hit ? (
          <p className="mt-3 text-xs text-quest-muted">
            高亮行：{hit[0] === hit[1] ? `L${hit[0]}` : `L${hit[0]}-L${hit[1]}`}
          </p>
        ) : null}
        <div
          className="source-viewer mt-4 overflow-auto rounded-2xl border border-quest-border/70 bg-[#fdfaf3]"
          data-source-path={payload.path}
          // shiki 的输出已 HTML 转义
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </Panel>
    </div>
  );
}

async function DirView({ dir }: { dir: string }) {
  let tree: SourceTreePayload;
  try {
    tree = await api.sourceTree(dir || undefined);
  } catch {
    notFound();
  }
  const title = tree.path ? tree.path : "仓库根";
  return (
    <Panel title={title} eyebrow="Source Tree / 目录浏览">
      <Breadcrumbs path={tree.path} isDir />
      <div className="mt-4">
        <TreeView tree={tree} />
      </div>
    </Panel>
  );
}

export default async function SourcePage({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const hit = parseLineRange(params.lines);

  if (params.path) {
    return <FileView path={params.path} hit={hit} />;
  }
  return <DirView dir={params.dir ?? ""} />;
}
