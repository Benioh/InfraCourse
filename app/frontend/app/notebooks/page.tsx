import Link from "next/link";
import { api } from "@/lib/api";
import type { NotebookCell } from "@/lib/types";
import { CommandBlock, Panel } from "@/components/ui";
import { Markdown } from "@/components/markdown";
import RegisterPageContext from "@/components/register-page-context";
import { withPublicPrefix } from "@/lib/runtime";

function jupyterUrl(path: string) {
  const base = process.env.NEXT_PUBLIC_JUPYTER_BASE ?? "http://localhost:8888";
  return `${base.replace(/\/$/, "")}/lab/tree/${path}`;
}

function NotebookCellView({ cell }: { cell: NotebookCell }) {
  if (cell.cell_type === "markdown") {
    return (
      <article className="rounded-3xl border border-quest-border bg-white/80 p-5">
        <Markdown source={cell.source} />
      </article>
    );
  }

  if (cell.cell_type === "code") {
    return (
      <article className="rounded-3xl border border-quest-border bg-white/80 p-5">
        <div className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-quest-muted">
          Code Cell {cell.execution_count ? `In [${cell.execution_count}]` : "未执行"}
        </div>
        <CommandBlock command={cell.source.trimEnd()} />
        {(cell.outputs ?? []).length ? (
          <div className="mt-4 space-y-3">
            {(cell.outputs ?? []).map((output, index) =>
              output.type === "image/png" ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  key={index}
                  alt="notebook output"
                  className="rounded-2xl border border-quest-border"
                  src={`data:image/png;base64,${output.body}`}
                />
              ) : (
                <pre
                  key={index}
                  className="overflow-x-auto whitespace-pre-wrap rounded-2xl border border-quest-border bg-[#f8f4ec] p-4 text-xs text-quest-ink"
                >
                  {output.body}
                </pre>
              ),
            )}
          </div>
        ) : null}
      </article>
    );
  }

  return (
    <article className="rounded-3xl border border-quest-border bg-white/80 p-5">
      <pre className="whitespace-pre-wrap text-sm text-quest-muted">{cell.source}</pre>
    </article>
  );
}

export default async function NotebookPage({
  searchParams,
}: {
  searchParams: Promise<{ path?: string }>;
}) {
  const { path } = await searchParams;
  if (!path) {
    return (
      <Panel title="Notebook 静态展示" eyebrow="Jupyter Concept Lab">
        <p className="text-sm text-quest-muted">
          缺少 notebook 路径，请从 mission 页面进入。
        </p>
      </Panel>
    );
  }

  const notebook = await api.notebook(path);
  const labUrl = jupyterUrl(notebook.path);

  // First markdown cell as a summary; falls back to title.
  const firstMarkdown = notebook.cells.find((c) => c.cell_type === "markdown");
  const summary = [
    `notebook=${notebook.path}`,
    `title=${notebook.title}`,
    firstMarkdown ? `intro=${firstMarkdown.source.slice(0, 240)}` : "",
  ]
    .filter(Boolean)
    .join(" | ");

  return (
    <div className="space-y-6" data-ai-context-root>
      <RegisterPageContext
        ctx={{
          page_kind: "notebook",
          notebook_path: notebook.path,
          page_summary: summary,
        }}
      />
      <Panel title={notebook.title} eyebrow={notebook.path}>
        <div className="flex flex-col gap-3 text-sm text-quest-muted md:flex-row md:items-center md:justify-between">
          <p>
            静态展示共 {notebook.cell_count} 个 cell。这里不会执行代码；需要运行请打开
            JupyterLab。
          </p>
          <div className="flex flex-wrap gap-2">
            <Link
              href={labUrl}
              target="_blank"
              className="rounded-full border border-quest-accent bg-quest-accent px-4 py-2 font-medium text-white hover:opacity-90"
            >
              在 JupyterLab 打开
            </Link>
            <Link
              href={withPublicPrefix("/")}
              className="rounded-full border border-quest-border bg-white/80 px-4 py-2 font-medium text-quest-ink hover:border-quest-accent"
            >
              回到任务总览
            </Link>
          </div>
        </div>
      </Panel>

      <div className="space-y-4">
        {notebook.cells.map((cell) => (
          <NotebookCellView key={cell.index} cell={cell} />
        ))}
      </div>
    </div>
  );
}
