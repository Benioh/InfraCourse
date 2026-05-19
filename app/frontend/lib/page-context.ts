// Lightweight page-context store. Pages mount <RegisterPageContext data={...} />
// to declare what the learner is currently looking at; the AI tutor reads this
// (alongside any text selection / URL params) when building the system prompt
// so questions like "这是干嘛的" don't need a manual file path.

export type PageContext = {
  /** Page archetype: mission | source | notebook | ticket | curriculum | ... */
  page_kind?: string;
  mission_id?: string;
  source_path?: string;
  lines?: [number, number];
  notebook_path?: string;
  ticket_id?: string;
  /** Free-form summary the page wants the tutor to know. */
  page_summary?: string;
  /** Trimmed innerText snapshot of the visible main region (set lazily). */
  visible_text?: string;
};

const KEY = "__INFRAQUEST_PAGE_CTX__";
const EVT = "infraquest:page-context";

declare global {
  interface Window {
    [KEY]?: PageContext | null;
  }
}

export function setPageContext(ctx: PageContext | null): void {
  if (typeof window === "undefined") return;
  window[KEY] = ctx;
  window.dispatchEvent(new CustomEvent(EVT, { detail: ctx }));
}

export function getPageContext(): PageContext | null {
  if (typeof window === "undefined") return null;
  return window[KEY] ?? null;
}

export function subscribePageContext(cb: (ctx: PageContext | null) => void): () => void {
  if (typeof window === "undefined") return () => {};
  const handler = (ev: Event) => cb((ev as CustomEvent<PageContext | null>).detail ?? null);
  window.addEventListener(EVT, handler);
  return () => window.removeEventListener(EVT, handler);
}

export const PAGE_CONTEXT_EVENT = EVT;
