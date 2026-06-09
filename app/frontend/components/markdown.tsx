"use client";

import type { ReactNode } from "react";
import { withPublicPrefix } from "@/lib/runtime";

function isAbsoluteImageSrc(src: string): boolean {
  return /^(?:https?:|data:|blob:)/i.test(src);
}

function normaliseRepoPath(value: string): string {
  const stack: string[] = [];
  for (const part of value.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") {
      stack.pop();
      continue;
    }
    stack.push(part);
  }
  return stack.join("/");
}

function imageSource(src: string, basePath?: string): string {
  const cleaned = src.trim();
  if (isAbsoluteImageSrc(cleaned)) return cleaned;
  if (cleaned.startsWith("/")) return withPublicPrefix(cleaned);

  const baseDir = basePath?.includes("/")
    ? basePath.slice(0, basePath.lastIndexOf("/"))
    : "";
  const repoPath = normaliseRepoPath(baseDir ? `${baseDir}/${cleaned}` : cleaned);
  return `${withPublicPrefix("/api/assets")}?path=${encodeURIComponent(repoPath)}`;
}

function renderInline(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const tokenPattern = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\)|https?:\/\/[^\s)]+)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(<span key={`text-${lastIndex}`}>{text.slice(lastIndex, match.index)}</span>);
    }
    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      nodes.push(
        <code
          key={`code-${match.index}`}
          className="rounded bg-[#eee4d7] px-1 py-0.5 font-mono text-[0.9em] text-quest-ink"
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith("**") && token.endsWith("**")) {
      nodes.push(
        <strong key={`strong-${match.index}`} className="font-semibold text-quest-ink">
          {renderInline(token.slice(2, -2))}
        </strong>,
      );
    } else {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      const href = linkMatch ? linkMatch[2] : token;
      const label = linkMatch ? linkMatch[1] : token;
      nodes.push(
        <a
          key={`link-${match.index}`}
          href={href}
          target="_blank"
          rel="noreferrer"
          className="font-medium text-quest-accent underline decoration-quest-accent/30 underline-offset-4 hover:decoration-quest-accent"
        >
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
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function isTableRow(line: string) {
  const t = line.trim();
  return t.startsWith("|") && t.endsWith("|") && t.includes("|", 1);
}

function isTableSeparator(line: string) {
  return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line.trim());
}

export type MarkdownProps = {
  source: string;
  /** When true, use a tighter visual style suitable for chat bubbles. */
  compact?: boolean;
  /** Extra classes on the wrapping <div>. */
  className?: string;
  /** Repository path of the markdown file; used to resolve relative image links. */
  basePath?: string;
};

export function Markdown({ source, compact = false, className, basePath }: MarkdownProps) {
  const lines = source.split("\n");
  const elements: ReactNode[] = [];
  let listItems: string[] = [];
  let listType: "ul" | "ol" = "ul";
  let paragraph: string[] = [];
  let quote: string[] = [];
  let codeLines: string[] | null = null;
  let codeLanguage = "";

  const pCls = compact
    ? "my-2 text-[13px] leading-6 text-quest-ink"
    : "my-3 text-sm leading-7 text-quest-muted";
  const ulCls = compact
    ? "my-2 list-disc space-y-0.5 pl-5 text-[13px] leading-6 text-quest-ink"
    : "my-3 list-disc space-y-1 pl-6 text-sm leading-7 text-quest-muted";
  const olCls = compact
    ? "my-2 list-decimal space-y-0.5 pl-5 text-[13px] leading-6 text-quest-ink"
    : "my-3 list-decimal space-y-1 pl-6 text-sm leading-7 text-quest-muted";
  const quoteCls = compact
    ? "my-3 border-l-2 border-quest-accent bg-[#fff8ec]/60 px-3 py-2 text-[13px] leading-6 text-quest-ink"
    : "my-4 rounded-2xl border-l-4 border-quest-accent bg-[#fff8ec] px-4 py-3 text-sm leading-7 text-quest-muted";
  const codeWrapCls = compact
    ? "my-3 overflow-hidden rounded-xl border border-quest-border bg-[#1f2933]"
    : "my-4 overflow-hidden rounded-2xl border border-quest-border bg-[#1f2933]";
  const imageCls = compact
    ? "my-3 w-full rounded-xl border border-quest-border bg-white object-cover"
    : "my-4 w-full rounded-2xl border border-quest-border bg-white object-cover";

  function flushList() {
    if (!listItems.length) return;
    if (listType === "ol") {
      elements.push(
        <ol key={`list-${elements.length}`} className={olCls}>
          {listItems.map((item, i) => (
            <li key={`${i}-${item}`}>{renderInline(item)}</li>
          ))}
        </ol>,
      );
    } else {
      elements.push(
        <ul key={`list-${elements.length}`} className={ulCls}>
          {listItems.map((item, i) => (
            <li key={`${i}-${item}`}>{renderInline(item)}</li>
          ))}
        </ul>,
      );
    }
    listItems = [];
  }

  function flushParagraph() {
    if (!paragraph.length) return;
    elements.push(
      <p key={`p-${elements.length}`} className={pCls}>
        {renderInline(paragraph.join(" "))}
      </p>,
    );
    paragraph = [];
  }

  function flushQuote() {
    if (!quote.length) return;
    elements.push(
      <blockquote key={`q-${elements.length}`} className={quoteCls}>
        {renderInline(quote.join(" "))}
      </blockquote>,
    );
    quote = [];
  }

  function flushCodeBlock() {
    if (!codeLines) return;
    elements.push(
      <div key={`fence-${elements.length}`} className={codeWrapCls}>
        {codeLanguage ? (
          <div className="border-b border-white/10 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-white/60">
            {codeLanguage}
          </div>
        ) : null}
        <pre className="overflow-x-auto p-3 text-[12px] leading-6 text-[#f8f4ec]">
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
      <div
        key={`table-${elements.length}`}
        className="my-4 overflow-x-auto rounded-2xl border border-quest-border bg-white"
      >
        <table className="min-w-full divide-y divide-quest-border text-left text-sm">
          <thead className="bg-[#f8f4ec] text-quest-ink">
            <tr>
              {header.map((cell, i) => (
                <th key={`${i}-${cell}`} className="px-4 py-3 font-semibold">
                  {renderInline(cell)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-quest-border text-quest-muted">
            {body.map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) => (
                  <td key={`${ri}-${ci}`} className="px-4 py-3 align-top">
                    {renderInline(cell)}
                  </td>
                ))}
              </tr>
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

    const imageMatch = trimmed.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (imageMatch) {
      flushList();
      flushParagraph();
      flushQuote();
      elements.push(
        <img
          key={`img-${elements.length}`}
          src={imageSource(imageMatch[2], basePath)}
          alt={imageMatch[1]}
          className={imageCls}
          loading="lazy"
        />,
      );
      continue;
    }

    if (isTableRow(line) && lines[index + 1] && isTableSeparator(lines[index + 1])) {
      const rows = [line, lines[index + 1]];
      index += 2;
      while (index < lines.length && isTableRow(lines[index])) {
        rows.push(lines[index]);
        index += 1;
      }
      index -= 1;
      flushList();
      flushParagraph();
      flushQuote();
      flushTable(rows);
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
      elements.push(
        <h3
          key={`h3-${elements.length}`}
          className={
            compact
              ? "mt-3 text-[13px] font-semibold text-quest-ink"
              : "mt-5 text-base font-semibold text-quest-ink"
          }
        >
          {renderInline(trimmed.slice(4))}
        </h3>,
      );
      continue;
    }
    if (trimmed.startsWith("## ")) {
      flushList();
      flushParagraph();
      flushQuote();
      elements.push(
        <h2
          key={`h2-${elements.length}`}
          className={
            compact
              ? "mt-4 font-mono text-sm font-semibold text-quest-ink"
              : "mt-6 font-mono text-lg font-semibold text-quest-ink"
          }
        >
          {renderInline(trimmed.slice(3))}
        </h2>,
      );
      continue;
    }
    if (trimmed.startsWith("# ")) {
      flushList();
      flushParagraph();
      flushQuote();
      elements.push(
        <h1
          key={`h1-${elements.length}`}
          className={
            compact
              ? "font-mono text-base font-semibold text-quest-ink"
              : "font-mono text-2xl font-semibold text-quest-ink"
          }
        >
          {renderInline(trimmed.slice(2))}
        </h1>,
      );
      continue;
    }
    if (trimmed.startsWith(">")) {
      flushList();
      flushParagraph();
      quote.push(trimmed.replace(/^>\s?/, ""));
      continue;
    }
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
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
  return <div className={className}>{elements}</div>;
}

export default Markdown;
