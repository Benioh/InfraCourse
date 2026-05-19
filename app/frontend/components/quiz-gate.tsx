"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { withPublicPrefix } from "@/lib/runtime";
import type {
  QuizPayload,
  QuizQuestionResult,
  QuizSubmissionResult,
} from "@/lib/types";

type Props = {
  missionId: string;
  onPassed?: () => void;
};

const KIND_LABELS: Record<string, string> = {
  read: "📄 必读",
  notebook: "📓 跑 Notebook",
  source: "🔍 读源码",
};

function PrereqLink({
  item,
}: {
  item: { title: string; kind: string; path: string; estimate_min?: number };
}) {
  let href = "#";
  if (item.kind === "notebook") {
    href = withPublicPrefix(`/notebooks?path=${encodeURIComponent(item.path)}`);
  } else if (item.kind === "source" || item.kind === "read") {
    href = withPublicPrefix(`/source?path=${encodeURIComponent(item.path)}`);
  }
  return (
    <li className="rounded-2xl border border-quest-border/70 bg-white/70 p-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm">{item.title}</span>
        <span className="rounded-full bg-[#f8f4ec] px-2 py-0.5 text-xs text-quest-muted">
          {KIND_LABELS[item.kind] ?? item.kind}
          {item.estimate_min ? ` · ~${item.estimate_min}min` : ""}
        </span>
      </div>
      <Link
        href={href}
        className="mt-2 inline-block font-mono text-xs text-quest-accent hover:underline"
      >
        {item.path}
      </Link>
    </li>
  );
}

export default function QuizGate({ missionId, onPassed }: Props) {
  const [quiz, setQuiz] = useState<QuizPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<QuizSubmissionResult | null>(null);
  const [opened, setOpened] = useState(false);

  useEffect(() => {
    api
      .quiz(missionId)
      .then(setQuiz)
      .catch((e) => setError(String(e)));
  }, [missionId]);

  if (error) {
    return (
      <div className="rounded-3xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        本关 quiz.yaml 还没配置（{error}）。可以直接进入 patch，但建议作者尽快补上 quiz。
      </div>
    );
  }
  if (!quiz) {
    return (
      <div className="rounded-3xl border border-quest-border/70 bg-white/70 p-4 text-sm text-quest-muted">
        加载 quiz…
      </div>
    );
  }

  const passed = quiz.status?.passed || result?.passed;
  const resultsById = new Map<string, QuizQuestionResult>(
    (result?.results ?? []).map((r) => [r.id, r])
  );

  const onSubmit = async () => {
    setSubmitting(true);
    setResult(null);
    try {
      const r = await api.submitQuiz(missionId, answers);
      setResult(r);
      if (r.passed) onPassed?.();
    } finally {
      setSubmitting(false);
    }
  };

  const allAnswered = quiz.questions.every((q) => answers[q.id]);

  return (
    <div className="rounded-3xl border border-quest-border/70 bg-white/80 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold">
            学习闭环 · 写代码前的必做事项
          </h3>
          <p className="mt-1 text-sm text-quest-muted">
            通过 quiz 才算"学懂"，才解锁 Patch 任务。
            {quiz.title ? ` · ${quiz.title}` : ""}
          </p>
        </div>
        {passed ? (
          <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-semibold text-green-800">
            ✅ Quiz Passed · {quiz.status?.score ?? result?.score}/
            {quiz.status?.total ?? result?.total}
          </span>
        ) : (
          <span className="rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-800">
            ⏳ 未通过 · 必须全对才解锁
          </span>
        )}
      </div>

      {/* Prereq checklist */}
      <div className="mt-4">
        <h4 className="text-sm font-semibold text-quest-ink">
          ① 先把这些资料读完（顺序 = 推荐学习顺序）
        </h4>
        <ul className="mt-2 grid gap-2 lg:grid-cols-2">
          {(quiz.prereq_checklist ?? []).map((p, i) => (
            <PrereqLink key={i} item={p} />
          ))}
        </ul>
      </div>

      {/* Quiz */}
      <div className="mt-5">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-quest-ink">
            ② 通过 Quiz Gate（{quiz.questions.length} 题，
            {Math.round((quiz.pass_threshold ?? 1) * 100)}% 才过）
          </h4>
          {!opened && !passed ? (
            <button
              onClick={() => setOpened(true)}
              className="rounded-full bg-quest-accent px-4 py-1.5 text-sm font-medium text-white hover:opacity-90"
            >
              开始测验
            </button>
          ) : null}
        </div>

        {quiz.description && (opened || passed) ? (
          <p className="mt-2 whitespace-pre-wrap rounded-2xl bg-[#fefae0] p-3 text-xs text-quest-muted">
            {quiz.description.trim()}
          </p>
        ) : null}

        {(opened || passed) && (
          <div className="mt-4 space-y-4">
            {quiz.questions.map((q, idx) => {
              const r = resultsById.get(q.id);
              return (
                <div
                  key={q.id}
                  className={`rounded-2xl border p-4 ${
                    r?.correct === false
                      ? "border-rose-300 bg-rose-50"
                      : r?.correct === true
                      ? "border-green-300 bg-green-50"
                      : "border-quest-border bg-white/70"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-sm font-semibold">
                    {idx + 1}. {q.prompt}
                  </p>
                  <div className="mt-3 space-y-2">
                    {q.options.map((opt) => {
                      const checked = answers[q.id] === opt.id;
                      const isExpected = r?.expected === opt.id;
                      const isYours = r?.your_answer === opt.id;
                      return (
                        <label
                          key={opt.id}
                          className={`flex cursor-pointer items-start gap-2 rounded-xl border p-2 text-sm ${
                            checked
                              ? "border-quest-accent bg-quest-accent/5"
                              : "border-quest-border bg-white"
                          } ${
                            r && isExpected
                              ? "ring-2 ring-green-400"
                              : r && isYours && !r.correct
                              ? "ring-2 ring-rose-400"
                              : ""
                          }`}
                        >
                          <input
                            type="radio"
                            name={q.id}
                            value={opt.id}
                            checked={checked}
                            onChange={() =>
                              setAnswers((prev) => ({ ...prev, [q.id]: opt.id }))
                            }
                            className="mt-1"
                            disabled={Boolean(r)}
                          />
                          <span className="whitespace-pre-wrap">
                            <span className="font-mono text-xs text-quest-muted">
                              [{opt.id}]
                            </span>{" "}
                            {opt.text}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                  {r ? (
                    <div className="mt-3 rounded-xl bg-white/80 p-3 text-xs leading-6 text-quest-ink">
                      <p
                        className={`font-semibold ${
                          r.correct ? "text-green-700" : "text-rose-700"
                        }`}
                      >
                        {r.correct
                          ? `✅ 答对了（[${r.expected}]）`
                          : `❌ 答错了，你选了 [${r.your_answer}]，正确是 [${r.expected}]`}
                      </p>
                      <pre className="mt-2 whitespace-pre-wrap font-sans">
                        {r.explanation}
                      </pre>
                    </div>
                  ) : null}
                </div>
              );
            })}

            {!result?.passed && (
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs text-quest-muted">
                  {allAnswered
                    ? "全部答完，准备提交"
                    : `还有 ${
                        quiz.questions.length -
                        Object.keys(answers).length
                      } 题没答`}
                </p>
                <button
                  onClick={onSubmit}
                  disabled={!allAnswered || submitting}
                  className="rounded-full bg-quest-accent px-5 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-40"
                >
                  {submitting
                    ? "提交中…"
                    : result
                    ? "重答（清空选项再试）"
                    : "提交答案"}
                </button>
              </div>
            )}

            {result && !result.passed ? (
              <div className="rounded-2xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
                得分 {result.score}/{result.total} · 阈值{" "}
                {Math.round(result.threshold * 100)}%。把上面错题的解析认真读一遍，
                重新读对应资料后再来一次。点「重答」清空选项继续。
              </div>
            ) : null}

            {result?.passed ? (
              <div className="rounded-2xl border border-green-300 bg-green-50 p-3 text-sm text-green-900">
                🎉 Quiz Passed！下面的 Patch 任务现在可以做了。
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
