"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  streamAi,
  type AiContext,
  type AiHistoryMessage,
  type AiMode,
} from "@/lib/ai-stream";
import { withPublicPrefix } from "@/lib/runtime";
import { getPageContext, subscribePageContext, type PageContext } from "@/lib/page-context";
import { Markdown } from "./markdown";
import { readSelection } from "./use-selection";

type Provider = "claude" | "codex";

type ToolEvent = {
  name: string;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  tools: ToolEvent[];
  pending?: boolean;
  error?: string;
};

const PROVIDER_KEY = "infraquest:ai:provider";
const MODE_KEY = "infraquest:ai:mode";

// Pull a trimmed innerText snapshot from the main region so the tutor
// has the same paragraph the learner is reading without manual selection.
function snapshotVisibleText(): string {
  if (typeof document === "undefined") return "";
  const main = document.querySelector("main") ?? document.body;
  const raw = (main as HTMLElement).innerText ?? "";
  const collapsed = raw.replace(/\s+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  return collapsed.length > 4000 ? `${collapsed.slice(0, 4000)}…` : collapsed;
}

export default function AiTutor() {
  const [open, setOpen] = useState(false);
  const [provider, setProvider] = useState<Provider>("claude");
  const [mode, setMode] = useState<AiMode>("tutor");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [includeSelection, setIncludeSelection] = useState(true);
  const [includePage, setIncludePage] = useState(true);
  const [selectionPreview, setSelectionPreview] = useState<string>("");
  const [pageCtx, setPageCtx] = useState<PageContext | null>(null);
  const [streaming, setStreaming] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedProvider = window.localStorage.getItem(PROVIDER_KEY);
    if (storedProvider === "claude" || storedProvider === "codex") setProvider(storedProvider);
    const storedMode = window.localStorage.getItem(MODE_KEY);
    if (storedMode === "tutor" || storedMode === "coder") setMode(storedMode);
  }, []);

  useEffect(() => {
    setPageCtx(getPageContext());
    return subscribePageContext(setPageCtx);
  }, []);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => {
    if (!open) return;
    const sel = readSelection();
    if (sel.text) {
      const trimmed = sel.text.length > 240 ? `${sel.text.slice(0, 240)}…` : sel.text;
      setSelectionPreview(trimmed);
    } else {
      setSelectionPreview("");
    }
  }, [open]);

  const setProviderPersisted = (p: Provider) => {
    setProvider(p);
    if (typeof window !== "undefined") window.localStorage.setItem(PROVIDER_KEY, p);
  };

  const setModePersisted = (m: AiMode) => {
    setMode(m);
    if (typeof window !== "undefined") window.localStorage.setItem(MODE_KEY, m);
  };

  const buildContext = useCallback((): AiContext | undefined => {
    const ctx: AiContext = {};
    if (includeSelection) {
      const sel = readSelection();
      if (sel.text) ctx.selection = sel.text;
      if (sel.sourcePath) ctx.source_path = sel.sourcePath;
      if (sel.lines) ctx.lines = sel.lines;
    }
    if (includePage && pageCtx) {
      if (pageCtx.page_kind && !ctx.page_kind) ctx.page_kind = pageCtx.page_kind;
      if (pageCtx.mission_id && !ctx.mission_id) ctx.mission_id = pageCtx.mission_id;
      if (pageCtx.source_path && !ctx.source_path) ctx.source_path = pageCtx.source_path;
      if (pageCtx.lines && !ctx.lines) ctx.lines = pageCtx.lines;
      if (pageCtx.notebook_path) ctx.notebook_path = pageCtx.notebook_path;
      if (pageCtx.ticket_id) ctx.ticket_id = pageCtx.ticket_id;
      if (pageCtx.page_summary) ctx.page_summary = pageCtx.page_summary;
      if (!ctx.visible_text) {
        const snap = snapshotVisibleText();
        if (snap) ctx.visible_text = snap;
      }
    }
    if (!ctx.source_path) {
      const path = searchParams.get("path");
      if (path) {
        ctx.source_path = path;
        const linesRaw = searchParams.get("lines");
        if (linesRaw) {
          const m = linesRaw.match(/^L?(\d+)(?:-L?(\d+))?$/);
          if (m) {
            const a = Number.parseInt(m[1], 10);
            const b = m[2] ? Number.parseInt(m[2], 10) : a;
            if (a > 0 && b > 0) ctx.lines = [Math.min(a, b), Math.max(a, b)];
          }
        }
      }
    }
    return Object.keys(ctx).length ? ctx : undefined;
  }, [includePage, includeSelection, pageCtx, searchParams]);

  const send = useCallback(async () => {
    const message = input.trim();
    if (!message || streaming) return;
    setInput("");

    const userId = `u-${Date.now()}`;
    const assistantId = `a-${Date.now()}`;
    const history: AiHistoryMessage[] = messages
      .filter((m) => !m.pending && !m.error)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: message, tools: [] },
      { id: assistantId, role: "assistant", content: "", tools: [], pending: true },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;
    setStreaming(true);

    await streamAi(
      { provider, mode, message, history, context: buildContext() },
      (event) => {
        setMessages((prev) =>
          prev.map((m) => {
            if (m.id !== assistantId) return m;
            switch (event.type) {
              case "text_delta":
                return { ...m, content: m.content + event.delta };
              case "tool_use":
                return {
                  ...m,
                  tools: [...m.tools, { name: event.name, input: event.input }],
                };
              case "tool_result": {
                const tools = [...m.tools];
                for (let i = tools.length - 1; i >= 0; i--) {
                  if (tools[i].name === event.name && tools[i].output == null) {
                    tools[i] = { ...tools[i], output: event.output };
                    break;
                  }
                }
                if (event.name === "navigate_to_source") {
                  const url = (event.output as { url?: string })?.url;
                  if (url && typeof url === "string") router.push(withPublicPrefix(url));
                }
                if (
                  (event.name === "write_starter" || event.name === "run_patch_test") &&
                  typeof window !== "undefined"
                ) {
                  // Let the mission page know the starter / patch state changed.
                  window.dispatchEvent(
                    new CustomEvent("infraquest:starter-changed", {
                      detail: event.output,
                    }),
                  );
                }
                return { ...m, tools };
              }
              case "done":
                return { ...m, pending: false };
              case "error":
                return { ...m, pending: false, error: event.message };
              default:
                return m;
            }
          }),
        );
      },
      controller.signal,
    );

    setStreaming(false);
    abortRef.current = null;
  }, [buildContext, input, messages, mode, provider, router, streaming]);

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
    setMessages((prev) =>
      prev.map((m) => (m.pending ? { ...m, pending: false, error: "已中断" } : m)),
    );
  };

  const clearMessages = () => {
    if (streaming) return;
    setMessages([]);
  };

  return (
    <>
      {/* Floating launcher — only visible when the panel is closed so it
          can't overlap the panel's own send / close controls. */}
      {!open ? (
        <button
          type="button"
          onClick={() => setOpen(true)}
          aria-label="打开 AI 助手"
          className="fixed bottom-7 right-7 z-40 inline-flex items-center gap-2 rounded-full border border-quest-border bg-quest-accent px-5 py-3 text-[13px] font-medium text-white shadow-panel transition-all duration-150 hover:bg-[#5a6e64] hover:shadow-drawer focus:outline-none focus:ring-3 focus:ring-quest-accent/30"
        >
          <ChatIcon />
          AI 助手
        </button>
      ) : null}

      {open ? (
        <aside className="fixed inset-y-0 right-0 z-30 flex w-full flex-col border-l border-quest-border bg-quest-card shadow-drawer sm:w-[30rem]">
          <header className="flex items-start justify-between gap-4 border-b border-quest-border-soft px-6 py-5">
            <div className="min-w-0">
              <p className="text-[11px] font-medium uppercase tracking-eyebrow text-quest-muted">
                AI Tutor
              </p>
              <p className="mt-1.5 text-sm font-semibold text-quest-ink">
                {mode === "coder" ? "Vibe Coding · 写代码模式" : "框架源码答疑"}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={clearMessages}
                disabled={streaming || messages.length === 0}
                className="rounded-chip border border-quest-border bg-quest-card-soft px-2.5 py-1 text-[11px] text-quest-muted transition-colors hover:text-quest-ink-soft disabled:opacity-40"
                title="清空对话"
              >
                清空
              </button>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="关闭 AI 助手"
                className="rounded-full border border-quest-border bg-quest-card-soft p-1.5 text-quest-muted hover:text-quest-ink"
              >
                <CloseIcon />
              </button>
            </div>
          </header>

          <div className="flex flex-wrap items-center gap-2 border-b border-quest-border-soft bg-quest-card-soft px-6 py-3 text-[11px]">
            <div className="flex gap-1 rounded-chip border border-quest-border bg-quest-card p-0.5">
              {(["tutor", "coder"] as const).map((m) => {
                const active = mode === m;
                return (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setModePersisted(m)}
                    className={
                      "rounded-[0.5rem] px-2.5 py-1 font-medium transition-colors duration-150 " +
                      (active
                        ? "bg-quest-accent text-white shadow-panel-soft"
                        : "text-quest-muted hover:text-quest-ink-soft")
                    }
                    title={
                      m === "tutor"
                        ? "答疑模式：读源码 / 解释概念"
                        : "写代码模式：按 vibe-coding 协议分段写 starter，跑测试"
                    }
                  >
                    {m === "tutor" ? "答疑" : "写代码"}
                  </button>
                );
              })}
            </div>
            <div className="flex gap-1 rounded-chip border border-quest-border bg-quest-card p-0.5">
              {(["claude", "codex"] as const).map((p) => {
                const active = provider === p;
                return (
                  <button
                    key={p}
                    type="button"
                    onClick={() => setProviderPersisted(p)}
                    className={
                      "rounded-[0.5rem] px-2.5 py-1 font-medium transition-colors duration-150 " +
                      (active
                        ? "bg-quest-accent text-white shadow-panel-soft"
                        : "text-quest-muted hover:text-quest-ink-soft")
                    }
                  >
                    {p}
                  </button>
                );
              })}
            </div>
            {pageCtx ? (
              <span
                className="ml-auto truncate rounded-full border border-quest-border bg-quest-card px-2.5 py-1 font-mono text-[10.5px] text-quest-muted"
                title={JSON.stringify(pageCtx, null, 2)}
              >
                {pageCtx.mission_id
                  ? `mission · ${pageCtx.mission_id}`
                  : pageCtx.source_path
                    ? `src · ${pageCtx.source_path.split("/").slice(-2).join("/")}`
                    : pageCtx.notebook_path
                      ? `nb · ${pageCtx.notebook_path.split("/").slice(-1)[0]}`
                      : pageCtx.page_kind ?? "page"}
              </span>
            ) : null}
          </div>

          <div ref={scrollRef} className="flex-1 overflow-auto px-6 py-5">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-start justify-center gap-3 text-sm leading-relaxed text-quest-muted">
                {mode === "coder" ? (
                  <>
                    <p className="text-quest-ink-soft">写代码模式按 vibe coding 协议分段。</p>
                    <ul className="mt-2 space-y-1.5 text-[12.5px] text-quest-muted">
                      <li>· 先复述 patch contract（task.md）</li>
                      <li>· 我分 3 段写 starter，每段后问一道理解题</li>
                      <li>· 失败时按 hint ladder（L0→L4）逐级抛提示</li>
                      <li>· 中途可让我跑 patch-test 看红绿</li>
                    </ul>
                  </>
                ) : (
                  <>
                    <p className="text-quest-ink-soft">
                      选中页面上一段代码或概念，再问我"这是干嘛的"。
                    </p>
                    <p>
                      我可以读源码、grep、跳转——基于
                      <span className="mx-1 rounded-sm bg-quest-accent-soft px-1.5 py-0.5 font-mono text-[11px] text-quest-accent">
                        {provider}
                      </span>
                      做答。
                    </p>
                    <ul className="mt-2 space-y-1.5 text-[12.5px] text-quest-muted">
                      <li>· "这个 forward 为什么要 chunk"</li>
                      <li>· "找一下 ColumnParallelLinear 的定义"</li>
                      <li>· "带我去 GAE chunk 的实现"</li>
                    </ul>
                  </>
                )}
              </div>
            ) : (
              messages.map((m) => <MessageBubble key={m.id} msg={m} />)
            )}
          </div>

          {(selectionPreview || pageCtx) ? (
            <div className="space-y-2 border-t border-quest-border-soft bg-quest-card-soft px-6 py-3">
              <div className="flex flex-wrap items-center gap-3 text-[11px] uppercase tracking-eyebrow text-quest-muted">
                {pageCtx ? (
                  <label className="inline-flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={includePage}
                      onChange={(e) => setIncludePage(e.target.checked)}
                      className="h-3 w-3 accent-quest-accent"
                    />
                    附带本页内容
                  </label>
                ) : null}
                {selectionPreview ? (
                  <label className="inline-flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={includeSelection}
                      onChange={(e) => setIncludeSelection(e.target.checked)}
                      className="h-3 w-3 accent-quest-accent"
                    />
                    附带选中片段
                  </label>
                ) : null}
              </div>
              {selectionPreview ? (
                <pre className="max-h-20 overflow-auto whitespace-pre-wrap rounded-md border border-quest-border-soft bg-quest-card p-2.5 font-mono text-[11px] leading-relaxed text-quest-ink-soft">
                  {selectionPreview}
                </pre>
              ) : null}
            </div>
          ) : null}

          <form
            className="border-t border-quest-border-soft bg-quest-card px-6 py-4"
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder={
                mode === "coder"
                  ? "比如：'我打算先用 list 装 grad 再 all_reduce'  ⌘ / Ctrl + Enter 发送"
                  : "问问这段代码…  ⌘ / Ctrl + Enter 发送"
              }
              rows={3}
              className="quest-focus-ring w-full resize-none rounded-chip border border-quest-border bg-quest-card-soft p-3 text-sm leading-relaxed text-quest-ink placeholder:text-quest-muted disabled:opacity-60"
              disabled={streaming}
            />
            <div className="mt-3 flex items-center justify-between">
              <span className="inline-flex items-center gap-2 text-[11px] text-quest-muted">
                <span
                  className={
                    "h-1.5 w-1.5 rounded-full " +
                    (streaming ? "animate-pulse bg-quest-accent" : "bg-quest-border")
                  }
                  aria-hidden
                />
                {streaming ? "生成中" : "就绪"}
              </span>
              {streaming ? (
                <button
                  type="button"
                  onClick={stop}
                  className="rounded-chip border border-quest-border bg-quest-card-soft px-4 py-1.5 text-[13px] font-medium text-quest-ink-soft transition-colors duration-150 hover:border-quest-dust hover:text-quest-dust"
                >
                  中断
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="rounded-chip bg-quest-accent px-5 py-1.5 text-[13px] font-medium text-white shadow-panel-soft transition-all duration-150 hover:bg-[#5a6e64] disabled:cursor-not-allowed disabled:bg-quest-border disabled:text-quest-muted disabled:shadow-none"
                >
                  发送
                </button>
              )}
            </div>
          </form>
        </aside>
      ) : null}
    </>
  );
}

function ChatIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M2 4.5C2 3.67 2.67 3 3.5 3h9c.83 0 1.5.67 1.5 1.5v6c0 .83-.67 1.5-1.5 1.5H6l-3 2.5V12H3.5C2.67 12 2 11.33 2 10.5v-6Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
      <path
        d="M3 3l8 8M11 3l-8 8"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <div className={`mb-4 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={
          "max-w-[92%] rounded-2xl px-4 py-3 text-[13.5px] leading-relaxed shadow-panel-soft " +
          (isUser
            ? "bg-quest-accent text-white"
            : "border border-quest-border bg-quest-card-soft text-quest-ink")
        }
      >
        {msg.tools.length > 0 ? (
          <div className="mb-2.5 flex flex-wrap gap-1.5">
            {msg.tools.map((t, i) => (
              <ToolChip key={i} tool={t} onUserBubble={isUser} />
            ))}
          </div>
        ) : null}
        {msg.content ? (
          isUser ? (
            <div className="whitespace-pre-wrap break-words">{msg.content}</div>
          ) : (
            <Markdown source={msg.content} compact className="ai-tutor-md" />
          )
        ) : msg.pending ? (
          <span className="inline-flex items-center gap-1.5 text-quest-muted">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-quest-muted [animation-delay:-0.2s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-quest-muted [animation-delay:-0.1s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-quest-muted" />
          </span>
        ) : null}
        {msg.error ? (
          <p className="mt-2 text-[12px] text-quest-dust">⚠ {msg.error}</p>
        ) : null}
      </div>
    </div>
  );
}

function ToolChip({ tool, onUserBubble }: { tool: ToolEvent; onUserBubble: boolean }) {
  const summary = summarize(tool);
  const tone = onUserBubble
    ? "border-white/30 bg-white/15 text-white"
    : "border-quest-border bg-quest-card text-quest-muted";
  return (
    <span
      className={
        "inline-flex max-w-full items-center gap-1.5 truncate rounded-full border px-2.5 py-0.5 font-mono text-[11px] " +
        tone
      }
    >
      <span className="opacity-70">›</span>
      <span className={onUserBubble ? "text-white" : "text-quest-ink-soft"}>{tool.name}</span>
      {summary ? <span className="truncate opacity-80">{summary}</span> : null}
    </span>
  );
}

function summarize(t: ToolEvent): string {
  const inp = t.input ?? {};
  switch (t.name) {
    case "navigate_to_source": {
      const path = String(inp.path ?? "");
      const lines = inp.lines as [number, number] | undefined;
      return lines ? `${path}#L${lines[0]}-${lines[1]}` : path;
    }
    case "read_file": {
      const path = String(inp.path ?? "");
      const a = inp.line_start as number | undefined;
      const b = inp.line_end as number | undefined;
      return a && b ? `${path}#L${a}-${b}` : path;
    }
    case "search_code": {
      const q = String(inp.query ?? "");
      const out = (t.output as { count?: number } | undefined) ?? {};
      return out.count != null ? `"${q}" → ${out.count} hits` : `"${q}"`;
    }
    case "write_starter": {
      const mission = String(inp.mission ?? "");
      const stage = inp.stage as string | undefined;
      return stage ? `${mission} · stage ${stage}` : mission;
    }
    case "run_patch_test": {
      const mission = String(inp.mission ?? "");
      const out = (t.output as { passed?: boolean } | undefined) ?? {};
      return out.passed != null ? `${mission} · ${out.passed ? "PASS" : "FAIL"}` : mission;
    }
    default:
      return "";
  }
}
