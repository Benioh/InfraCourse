import Link from "next/link";
import type { ReactNode } from "react";
import { api } from "@/lib/api";
import type { NotebookCell } from "@/lib/types";
import { CommandBlock, Panel } from "@/components/ui";

function jupyterUrl(path: string) {
  const base = process.env.NEXT_PUBLIC_JUPYTER_BASE ?? "http://localhost:8888";
  return `${base.replace(/\/$/, "")}/lab/tree/${path}`;
}

function renderInline(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const tokenPattern = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\(https?:\/\/[^)]+\)|https?:\/\/[^\s)]+)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(<span key={`text-${lastIndex}`}>{text.slice(lastIndex, match.index)}</span>);
    }

    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      nodes.push(
        <code key={`code-${match.index}`} className="rounded bg-[#eee4d7] px-1 py-0.5 font-mono text-sm text-quest-ink">
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith("**") && token.endsWith("**")) {
      nodes.push(<strong key={`strong-${match.index}`} className="font-semibold text-quest-ink">{renderInline(token.slice(2, -2))}</strong>);
    } else {
      const linkMatch = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/);
      const href = linkMatch ? linkMatch[2] : token;
      const label = linkMatch ? linkMatch[1] : token;
      nodes.push(
        <a key={`link-${match.index}`} href={href} target="_blank" rel="noreferrer" className="font-medium text-quest-accent underline decoration-quest-accent/30 underline-offset-4 hover:decoration-quest-accent">
          {label}
        </a>,
      );
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    nodes.push(<span key={`text-${lastIndex}`}>{text.slice(lastIndex)}</span>);
  }

  return nodes;
}

function splitTableRow(line: string) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isTableRow(line: string) {
  const trimmed = line.trim();
  return trimmed.startsWith("|") && trimmed.endsWith("|") && trimmed.includes("|", 1);
}

function isTableSeparator(line: string) {
  return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line.trim());
}

function MarkdownBlock({ source }: { source: string }) {
  const lines = source.split("\n");
  const elements: ReactNode[] = [];
  let listItems: string[] = [];
  let listType: "ul" | "ol" = "ul";
  let paragraph: string[] = [];
  let quote: string[] = [];
  let codeLines: string[] | null = null;
  let codeLanguage = "";

  function flushList() {
    if (!listItems.length) return;
    const Tag = listType;
    elements.push(
      <Tag key={`list-${elements.length}`} className={`${listType === "ol" ? "list-decimal" : "list-disc"} my-3 space-y-1 pl-6 text-sm leading-7 text-quest-muted`}>
        {listItems.map((item, index) => <li key={`${index}-${item}`}>{renderInline(item)}</li>)}
      </Tag>,
    );
    listItems = [];
  }

  function flushParagraph() {
    if (!paragraph.length) return;
    elements.push(
      <p key={`p-${elements.length}`} className="my-3 text-sm leading-7 text-quest-muted">
        {renderInline(paragraph.join(" "))}
      </p>,
    );
    paragraph = [];
  }

  function flushQuote() {
    if (!quote.length) return;
    elements.push(
      <blockquote key={`quote-${elements.length}`} className="my-4 rounded-2xl border-l-4 border-quest-accent bg-[#fff8ec] px-4 py-3 text-sm leading-7 text-quest-muted">
        {renderInline(quote.join(" "))}
      </blockquote>,
    );
    quote = [];
  }

  function flushCodeBlock() {
    if (!codeLines) return;
    elements.push(
      <div key={`fence-${elements.length}`} className="my-4 overflow-hidden rounded-2xl border border-quest-border bg-[#1f2933]">
        {codeLanguage ? <div className="border-b border-white/10 px-4 py-2 font-mono text-xs uppercase tracking-[0.16em] text-white/60">{codeLanguage}</div> : null}
        <pre className="overflow-x-auto p-4 text-xs leading-6 text-[#f8f4ec]">
          <code>{codeLines.join("\n")}</code>
        </pre>
      </div>,
    );
    codeLines = null;
    codeLanguage = "";
  }

  function flushTable(rows: string[]) {
    if (rows.length < 2) return;
    const header = splitTableRow(rows[0]);
    const body = rows.slice(2).map(splitTableRow);
    elements.push(
      <div key={`table-${elements.length}`} className="my-4 overflow-x-auto rounded-2xl border border-quest-border bg-white">
        <table className="min-w-full divide-y divide-quest-border text-left text-sm">
          <thead className="bg-[#f8f4ec] text-quest-ink">
            <tr>{header.map((cell, index) => <th key={`${index}-${cell}`} className="px-4 py-3 font-semibold">{renderInline(cell)}</th>)}</tr>
          </thead>
          <tbody className="divide-y divide-quest-border text-quest-muted">
            {body.map((row, rowIndex) => (
              <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={`${rowIndex}-${cellIndex}`} className="px-4 py-3 align-top">{renderInline(cell)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>,
    );
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (codeLines) {
      if (trimmed.startsWith("```")) {
        flushCodeBlock();
      } else {
        codeLines.push(line);
      }
      continue;
    }

    if (trimmed.startsWith("```")) {
      flushList();
      flushParagraph();
      flushQuote();
      codeLanguage = trimmed.slice(3).trim();
      codeLines = [];
      continue;
    }

    if (!trimmed) {
      flushList();
      flushParagraph();
      flushQuote();
      continue;
    }

    if (isTableRow(line) && lines[index + 1] && isTableSeparator(lines[index + 1])) {
      const tableRows = [line, lines[index + 1]];
      index += 2;
      while (index < lines.length && isTableRow(lines[index])) {
        tableRows.push(lines[index]);
        index += 1;
      }
      index -= 1;
      flushList();
      flushParagraph();
      flushQuote();
      flushTable(tableRows);
      continue;
    }

    if (/^-{3,}$/.test(trimmed)) {
      flushList();
      flushParagraph();
      flushQuote();
      elements.push(<hr key={`hr-${elements.length}`} className="my-5 border-quest-border" />);
      continue;
    }

    if (trimmed.startsWith("### ")) {
      flushList();
      flushParagraph();
      flushQuote();
      elements.push(<h3 key={`h3-${elements.length}`} className="mt-5 text-base font-semibold text-quest-ink">{renderInline(trimmed.slice(4))}</h3>);
      continue;
    }
    if (trimmed.startsWith("## ")) {
      flushList();
      flushParagraph();
      flushQuote();
      elements.push(<h2 key={`h2-${elements.length}`} className="mt-6 font-mono text-lg font-semibold text-quest-ink">{renderInline(trimmed.slice(3))}</h2>);
      continue;
    }
    if (trimmed.startsWith("# ")) {
      flushList();
      flushParagraph();
      flushQuote();
      elements.push(<h1 key={`h1-${elements.length}`} className="font-mono text-2xl font-semibold text-quest-ink">{renderInline(trimmed.slice(2))}</h1>);
      continue;
    }
    if (trimmed.startsWith(">")) {
      flushList();
      flushParagraph();
      quote.push(trimmed.replace(/^>\s?/, ""));
      continue;
    }
    if (trimmed.startsWith("- ")) {
      flushParagraph();
      flushQuote();
      if (listType !== "ul") {
        flushList();
        listType = "ul";
      }
      listType = "ul";
      listItems.push(trimmed.slice(2));
      continue;
    }
    const ordered = trimmed.match(/^\d+\.\s+(.*)$/);
    if (ordered) {
      flushParagraph();
      flushQuote();
      if (listType !== "ol") {
        flushList();
        listType = "ol";
      }
      listType = "ol";
      listItems.push(ordered[1]);
      continue;
    }
    paragraph.push(trimmed);
  }
  flushCodeBlock();
  flushList();
  flushParagraph();
  flushQuote();
  return <div>{elements}</div>;
}

function NotebookCellView({ cell }: { cell: NotebookCell }) {
  if (cell.cell_type === "markdown") {
    return (
      <article className="rounded-3xl border border-quest-border bg-white/80 p-5">
        <MarkdownBlock source={cell.source} />
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
                <img key={index} alt="notebook output" className="rounded-2xl border border-quest-border" src={`data:image/png;base64,${output.body}`} />
              ) : (
                <pre key={index} className="overflow-x-auto whitespace-pre-wrap rounded-2xl border border-quest-border bg-[#f8f4ec] p-4 text-xs text-quest-ink">
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

export default async function NotebookPage({ searchParams }: { searchParams: Promise<{ path?: string }> }) {
  const { path } = await searchParams;
  if (!path) {
    return (
      <Panel title="Notebook 静态展示" eyebrow="Jupyter Concept Lab">
        <p className="text-sm text-quest-muted">缺少 notebook 路径，请从 mission 页面进入。</p>
      </Panel>
    );
  }

  const notebook = await api.notebook(path);
  const labUrl = jupyterUrl(notebook.path);

  return (
    <div className="space-y-6">
      <Panel title={notebook.title} eyebrow={notebook.path}>
        <div className="flex flex-col gap-3 text-sm text-quest-muted md:flex-row md:items-center md:justify-between">
          <p>静态展示共 {notebook.cell_count} 个 cell。这里不会执行代码；需要运行请打开 JupyterLab。</p>
          <div className="flex flex-wrap gap-2">
            <Link href={labUrl} target="_blank" className="rounded-full border border-quest-accent bg-quest-accent px-4 py-2 font-medium text-white hover:opacity-90">
              在 JupyterLab 打开
            </Link>
            <Link href="/" className="rounded-full border border-quest-border bg-white/80 px-4 py-2 font-medium text-quest-ink hover:border-quest-accent">
              回到任务总览
            </Link>
          </div>
        </div>
      </Panel>

      <div className="space-y-4">
        {notebook.cells.map((cell) => <NotebookCellView key={cell.index} cell={cell} />)}
      </div>
    </div>
  );
}
