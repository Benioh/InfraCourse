import type {
  BadgeRow,
  CurriculumPayload,
  DashboardPayload,
  NotebookPayload,
  PatchPayload,
  PatchStatus,
  Quest,
  QuizPayload,
  QuizStatus,
  QuizSubmissionResult,
  SourcePayload,
  SourceTreePayload,
  Ticket,
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!response.ok) {
    throw new Error(`请求失败：${path}，状态码 ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  dashboard: () => fetchJson<DashboardPayload>("/dashboard"),
  curriculum: () => fetchJson<CurriculumPayload>("/curriculum"),
  quests: () => fetchJson<Quest[]>("/quests"),
  quest: (missionId: string) => fetchJson<Quest>(`/quests/${missionId}`),

  // ★ Quiz Gate endpoints (must pass before patch)
  quiz: (missionId: string) => fetchJson<QuizPayload>(`/missions/${missionId}/quiz`),
  quizStatus: (missionId: string) =>
    fetchJson<QuizStatus>(`/missions/${missionId}/quiz-status`),
  submitQuiz: (missionId: string, answers: Record<string, string>) =>
    fetchJson<QuizSubmissionResult>(`/missions/${missionId}/quiz`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers }),
    }),

  // ★ Patch Track endpoints
  patch: (missionId: string) => fetchJson<PatchPayload>(`/missions/${missionId}/patch`),
  patchStatus: (missionId: string) =>
    fetchJson<PatchStatus>(`/missions/${missionId}/patch-status`),
  runPatchTest: (missionId: string) =>
    fetchJson<PatchStatus>(`/missions/${missionId}/patch-test`, { method: "POST" }),

  tickets: () => fetchJson<Ticket[]>("/tickets"),
  ticket: (ticketId: string) => fetchJson<Ticket>(`/tickets/${ticketId}`),
  metrics: (mission?: string) =>
    fetchJson<Record<string, unknown>[]>(mission ? `/metrics?mission=${mission}` : "/metrics"),
  report: (mission: string) =>
    fetchJson<{ mission: string; source?: string; body: string }>(`/reports/${mission}`),
  badges: () => fetchJson<BadgeRow[]>("/badges"),
  prompts: () =>
    fetchJson<{ id: string; title: string; body: string; path: string }[]>("/prompts"),
  notebook: (path: string) =>
    fetchJson<NotebookPayload>(`/notebooks?path=${encodeURIComponent(path)}`),
  conceptMap: () =>
    fetchJson<
      {
        id: string;
        definition: string;
        why_it_matters: string;
        where_it_appears: string[];
        related_experiments: string[];
        common_failure: string;
      }[]
    >("/concept-map"),
  sourceMap: (framework: string) =>
    fetchJson<{
      framework: string;
      nodes: {
        node: string;
        what_to_search: string;
        what_to_inspect: string;
        question: string;
      }[];
    }>(`/source-map/${framework}`),
  source: (path: string) =>
    fetchJson<SourcePayload>(`/source?path=${encodeURIComponent(path)}`),
  sourceTree: (dir?: string) =>
    fetchJson<SourceTreePayload>(dir ? `/source/tree?dir=${encodeURIComponent(dir)}` : "/source/tree"),
};
