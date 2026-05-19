export function normalizePublicPrefix(value: string | undefined): string {
  if (!value) return "";
  const trimmed = value.trim().replace(/\/+$/, "");
  if (!trimmed) return "";
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

export const PROXY_PREFIX = normalizePublicPrefix(process.env.NEXT_PUBLIC_PROXY_PREFIX);
export const BASE_PATH = normalizePublicPrefix(process.env.NEXT_PUBLIC_BASE_PATH);
export const PUBLIC_PREFIX = PROXY_PREFIX || BASE_PATH;

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ??
  (typeof window === "undefined"
    ? `${process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8000"}/api`
    : `${PUBLIC_PREFIX}/api`);

export function withPublicPrefix(path: string): string {
  if (!PROXY_PREFIX || !path.startsWith("/") || path.startsWith("//")) {
    return path;
  }
  if (path === PROXY_PREFIX || path.startsWith(`${PROXY_PREFIX}/`) || path.startsWith(`${PROXY_PREFIX}?`)) {
    return path;
  }
  return `${PROXY_PREFIX}${path}`;
}
