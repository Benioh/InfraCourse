// SSE parser for /api/ai/ask. EventSource is GET-only so we use fetch + a
// ReadableStream reader and parse `data: <json>\n\n` frames manually.

import { API_BASE } from "@/lib/runtime";

export type AiEvent =
  | { type: "text_delta"; delta: string }
  | { type: "tool_use"; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; name: string; output: Record<string, unknown> }
  | { type: "done" }
  | { type: "error"; message: string };

export type AiContext = {
  selection?: string;
  source_path?: string;
  lines?: [number, number];
  mission_id?: string;
  /** Page archetype: mission | source | notebook | ticket | curriculum | ... */
  page_kind?: string;
  /** Notebook path when on a notebook page. */
  notebook_path?: string;
  /** Free-form summary registered by the page itself. */
  page_summary?: string;
  /** Trimmed innerText snapshot of the visible main region. */
  visible_text?: string;
};

export type AiHistoryMessage = { role: "user" | "assistant"; content: string };

export type AiMode = "tutor" | "coder";

export type AiAskBody = {
  provider: "claude" | "codex";
  message: string;
  /** Tutor (default) = answer questions. Coder = vibe-coding mode that
   * may call write_starter / run_patch_test. */
  mode?: AiMode;
  context?: AiContext;
  history?: AiHistoryMessage[];
};

export async function streamAi(
  body: AiAskBody,
  onEvent: (event: AiEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/ai/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    onEvent({
      type: "error",
      message: `HTTP ${response.status}${text ? `: ${text.slice(0, 300)}` : ""}`,
    });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line (\n\n). Split, keep tail.
      let frameEnd: number;
      while ((frameEnd = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, frameEnd);
        buffer = buffer.slice(frameEnd + 2);
        const event = parseFrame(frame);
        if (event) onEvent(event);
      }
    }
    if (buffer.trim()) {
      const event = parseFrame(buffer);
      if (event) onEvent(event);
    }
  } catch (err) {
    if ((err as { name?: string })?.name === "AbortError") return;
    onEvent({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
}

function parseFrame(frame: string): AiEvent | null {
  const lines = frame.split("\n");
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) return null;
  const payload = dataLines.join("\n");
  if (!payload) return null;
  try {
    return JSON.parse(payload) as AiEvent;
  } catch {
    return null;
  }
}
