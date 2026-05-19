"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { PatchStatus } from "@/lib/types";

type Props = {
  missionId: string;
  initialContent: string;
  path: string;
  language?: string;
  quizPassed?: boolean;
  initialPatchStatus?: PatchStatus | null;
};

function StatusBadge({ status }: { status: PatchStatus | null | undefined }) {
  if (!status || status.never_run) {
    return (
      <span className="rounded-full border border-quest-border px-3 py-1 text-xs text-quest-muted">
        从未运行
      </span>
    );
  }
  if (status.passed) {
    return (
      <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-semibold text-green-800">
        ✅ PASS · {status.summary?.passed ?? 0}/{status.summary?.total ?? 0}
        {status.duration_s != null ? ` · ${status.duration_s.toFixed(1)}s` : ""}
      </span>
    );
  }
  return (
    <span className="rounded-full bg-rose-100 px-3 py-1 text-xs font-semibold text-rose-800">
      ❌ FAIL · {status.summary?.failed ?? 0}/{status.summary?.total ?? 0} 失败
    </span>
  );
}

export default function CodeEditor({
  missionId,
  initialContent,
  path,
  language,
  quizPassed,
  initialPatchStatus,
}: Props) {
  const [content, setContent] = useState(initialContent);
  const [savedContent, setSavedContent] = useState(initialContent);
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<PatchStatus | null | undefined>(initialPatchStatus);
  const [error, setError] = useState<string | null>(null);
  const taRef = useRef<HTMLTextAreaElement | null>(null);

  const dirty = content !== savedContent;
  const gated = quizPassed === false;

  // Listen for AI-driven writes (write_starter / run_patch_test) and refresh.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const onChange = () => {
      api
        .starter(missionId)
        .then((s) => {
          setContent(s.content);
          setSavedContent(s.content);
        })
        .catch(() => {});
      api
        .patchStatus(missionId)
        .then(setStatus)
        .catch(() => {});
    };
    window.addEventListener("infraquest:starter-changed", onChange);
    return () => window.removeEventListener("infraquest:starter-changed", onChange);
  }, [missionId]);

  const onSave = useCallback(async () => {
    setSaving(true);
    setError(null);
    try {
      const result = await api.saveStarter(missionId, content);
      setSavedContent(content);
      // Server returns the new payload; if it included a content (it does),
      // sync just in case server normalised line endings.
      if ("content" in (result as Record<string, unknown>)) {
        const c = (result as { content?: string }).content;
        if (typeof c === "string") {
          setContent(c);
          setSavedContent(c);
        }
      }
    } catch (exc) {
      setError(String(exc));
    } finally {
      setSaving(false);
    }
  }, [content, missionId]);

  const onRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      if (dirty) {
        await api.saveStarter(missionId, content);
        setSavedContent(content);
      }
      const result = await api.runPatchTest(missionId);
      setStatus(result);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setRunning(false);
    }
  }, [content, dirty, missionId]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // ⌘/Ctrl+S → save.
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      void onSave();
      return;
    }
    // Tab → insert two spaces instead of changing focus.
    if (e.key === "Tab") {
      e.preventDefault();
      const ta = taRef.current;
      if (!ta) return;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      const next = content.slice(0, start) + "  " + content.slice(end);
      setContent(next);
      requestAnimationFrame(() => {
        if (taRef.current) {
          taRef.current.selectionStart = start + 2;
          taRef.current.selectionEnd = start + 2;
        }
      });
    }
  };

  const lineCount = content.split("\n").length;

  return (
    <div className="rounded-3xl border border-quest-border/70 bg-white/80 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold">
            {gated ? "🔒 写代码 · Patch" : "写代码 · Patch"}
          </h3>
          <StatusBadge status={status} />
          {dirty ? (
            <span className="rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-xs text-amber-800">
              未保存
            </span>
          ) : null}
          <span
            className="font-mono text-[11px] text-quest-muted"
            data-source-path={path}
          >
            {path} · {lineCount} 行
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onSave}
            disabled={saving || !dirty}
            className="rounded-full border border-quest-border px-3 py-1.5 text-sm text-quest-ink hover:border-quest-accent disabled:opacity-50"
            title="⌘/Ctrl+S"
          >
            {saving ? "保存中…" : "保存"}
          </button>
          <button
            type="button"
            onClick={onRun}
            disabled={running}
            className={`rounded-full px-4 py-1.5 text-sm font-medium hover:opacity-90 disabled:opacity-50 ${
              gated
                ? "border border-amber-400 bg-white text-amber-800"
                : "bg-quest-accent text-white"
            }`}
            title={gated ? "建议先通过 Quiz 再来跑 patch-test" : ""}
          >
            {running ? "测试中…" : dirty ? "保存并跑 patch-test" : "跑 patch-test"}
          </button>
        </div>
      </div>

      {gated ? (
        <p className="mt-3 rounded-2xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
          ⚠️ 你还没通过本关 Quiz Gate。强烈建议先把概念关吃透再写代码。
        </p>
      ) : null}

      <textarea
        ref={taRef}
        value={content}
        onChange={(e) => setContent(e.target.value)}
        onKeyDown={onKeyDown}
        spellCheck={false}
        className="mt-4 block max-h-[640px] min-h-[280px] w-full resize-y overflow-auto rounded-2xl border border-quest-border bg-[#fdfaf3] p-4 font-mono text-[12.5px] leading-6 text-quest-ink focus:border-quest-accent focus:outline-none"
        data-language={language ?? "text"}
      />

      {error ? (
        <pre className="mt-3 whitespace-pre-wrap rounded-2xl bg-rose-50 p-3 text-xs text-rose-700">
          {error}
        </pre>
      ) : null}

      {status?.output ? (
        <details className="mt-4" open={!status.passed}>
          <summary className="cursor-pointer text-sm text-quest-muted">
            Pytest 输出（{status.output_truncated ? "末尾 64KB" : "完整"}）
          </summary>
          <pre className="mt-2 max-h-[480px] overflow-auto whitespace-pre-wrap rounded-2xl bg-[#0f172a] p-4 font-mono text-xs leading-relaxed text-slate-100">
            {status.output}
          </pre>
        </details>
      ) : null}
    </div>
  );
}
