"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { PatchStatus } from "@/lib/types";

type Props = {
  missionId: string;
  initialStatus: PatchStatus | null | undefined;
  quizPassed?: boolean;
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
        ✅ PASS · {status.summary?.passed ?? 0}/{status.summary?.total ?? 0} ·{" "}
        {status.duration_s?.toFixed(1)}s
      </span>
    );
  }
  return (
    <span className="rounded-full bg-rose-100 px-3 py-1 text-xs font-semibold text-rose-800">
      ❌ FAIL · {status.summary?.failed ?? 0}/{status.summary?.total ?? 0} 失败
    </span>
  );
}

export default function PatchRunner({ missionId, initialStatus, quizPassed }: Props) {
  const [status, setStatus] = useState<PatchStatus | null | undefined>(initialStatus);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await api.runPatchTest(missionId);
      setStatus(result);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setRunning(false);
    }
  };

  const gated = quizPassed === false;

  return (
    <div
      className={`rounded-3xl border p-5 ${
        gated ? "border-amber-300 bg-amber-50/40" : "border-quest-border/70 bg-white/80"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold">
            {gated ? "🔒 Patch 任务" : "Patch 状态"}
          </h3>
          <StatusBadge status={status} />
          {status?.finished_at && (
            <span className="text-xs text-quest-muted">
              最近运行：{new Date(status.finished_at).toLocaleString()}
            </span>
          )}
        </div>
        <button
          onClick={onRun}
          disabled={running}
          className={`rounded-full px-4 py-2 text-sm font-medium hover:opacity-90 disabled:opacity-50 ${
            gated
              ? "border border-amber-400 bg-white text-amber-800"
              : "bg-quest-accent text-white"
          }`}
          title={gated ? "建议先通过上方 Quiz 再来跑 patch-test" : ""}
        >
          {running
            ? "测试运行中…"
            : gated
            ? "强制运行（不推荐）"
            : "运行 patch-test"}
        </button>
      </div>

      {gated && (
        <div className="mt-3 rounded-2xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
          ⚠️ 你还没通过本关 Quiz Gate（上方）。强烈建议先把概念关吃透再写代码——否则
          patch-test 通过了你也不一定真懂为什么。
        </div>
      )}

      {error && (
        <pre className="mt-3 whitespace-pre-wrap rounded-2xl bg-rose-50 p-3 text-xs text-rose-700">
          {error}
        </pre>
      )}

      {status?.output && (
        <details className="mt-4" open={!status.passed}>
          <summary className="cursor-pointer text-sm text-quest-muted">
            Pytest 输出（{status.output_truncated ? "末尾 64KB" : "完整"}）
          </summary>
          <pre className="mt-2 max-h-[480px] overflow-auto whitespace-pre-wrap rounded-2xl bg-[#0f172a] p-4 font-mono text-xs leading-relaxed text-slate-100">
            {status.output}
          </pre>
        </details>
      )}

      {!status?.output && !running && (
        <p className="mt-3 text-sm text-quest-muted">
          点击右上角按钮触发 <code>{`make patch-test M=${missionId}`}</code>。
          后端会跑 pytest 并把结果写到 <code>labs/{missionId}/patch/.last_run.json</code>，
          页面立即更新。
        </p>
      )}
    </div>
  );
}
