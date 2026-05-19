import { withPublicPrefix } from "@/lib/runtime";

export type ParsedPath = {
  path: string;
  hitRange: [number, number] | null;
};

export function parseSourceHref(raw: string): ParsedPath {
  const cleaned = raw.trim();
  if (!cleaned) return { path: "", hitRange: null };
  const hashIdx = cleaned.indexOf("#");
  if (hashIdx >= 0) {
    const before = cleaned.slice(0, hashIdx);
    const frag = cleaned.slice(hashIdx + 1);
    return { path: before, hitRange: parseLineRange(frag) };
  }
  return { path: cleaned, hitRange: null };
}

export function parseLineRange(raw: string | null | undefined): [number, number] | null {
  if (!raw) return null;
  let value = raw.trim();
  if (!value) return null;
  if (value.toLowerCase().startsWith("l")) value = value.slice(1);
  const dashIdx = value.indexOf("-");
  if (dashIdx < 0) {
    const single = Number.parseInt(value, 10);
    if (!Number.isFinite(single) || single <= 0) return null;
    return [single, single];
  }
  let endRaw = value.slice(dashIdx + 1).trim();
  if (endRaw.toLowerCase().startsWith("l")) endRaw = endRaw.slice(1);
  const start = Number.parseInt(value.slice(0, dashIdx), 10);
  const end = Number.parseInt(endRaw, 10);
  if (!Number.isFinite(start) || !Number.isFinite(end) || start <= 0 || end <= 0) return null;
  return start <= end ? [start, end] : [end, start];
}

export function sourceUrl(path: string, lines?: [number, number] | null): string {
  const query = new URLSearchParams({ path });
  if (lines) query.set("lines", lines[0] === lines[1] ? `${lines[0]}` : `${lines[0]}-${lines[1]}`);
  return withPublicPrefix(`/source?${query.toString()}`);
}

export function buildBreadcrumbs(path: string): { name: string; href: string; isLast: boolean }[] {
  if (!path) return [];
  const segments = path.split("/").filter(Boolean);
  const crumbs: { name: string; href: string; isLast: boolean }[] = [];
  let acc = "";
  segments.forEach((seg, idx) => {
    acc = acc ? `${acc}/${seg}` : seg;
    const isLast = idx === segments.length - 1;
    const href = isLast ? sourceUrl(acc) : withPublicPrefix(`/source?dir=${encodeURIComponent(acc)}`);
    crumbs.push({ name: seg, href, isLast });
  });
  return crumbs;
}

export function splitInspectPaths(raw: string): string[] {
  if (!raw) return [];
  return raw
    .split(/[;；]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function inferRepoPath(candidate: string): string | null {
  const trimmed = candidate.trim();
  if (!trimmed) return null;
  const stripped = trimmed.replace(/^[`'"]+|[`'"]+$/g, "");
  const hashIdx = stripped.indexOf("#");
  const withoutAnchor = hashIdx >= 0 ? stripped.slice(0, hashIdx) : stripped;
  const first = withoutAnchor.split("/")[0];
  const allowed = new Set([
    "github_repo", "labs", "scripts", "docs", "notebooks",
    "quests", "tickets", "prompts", "app", "autograder",
    "mini_infra", "simulators", "envs", "dashboards", "final_artifacts",
  ]);
  if (!allowed.has(first)) return null;
  return stripped;
}
