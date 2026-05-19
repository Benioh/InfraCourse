import type { NextConfig } from "next";

function normalizeBasePath(value: string | undefined): string {
  if (!value) return "";
  const trimmed = value.trim().replace(/\/+$/, "");
  if (!trimmed) return "";
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

const basePath = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);
const proxyPrefix = normalizeBasePath(process.env.NEXT_PUBLIC_PROXY_PREFIX);
const backendOrigin = process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8000";
const explicitAssetPrefix = process.env.NEXT_PUBLIC_ASSET_PREFIX?.trim().replace(/\/+$/, "");
const assetPrefix = explicitAssetPrefix ?? (proxyPrefix || basePath);

const nextConfig: NextConfig = {
  typedRoutes: false,
  ...(basePath && !proxyPrefix ? { basePath } : {}),
  ...(assetPrefix ? { assetPrefix } : {}),
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendOrigin}/api/:path*`,
        basePath: false,
      },
    ];
  },
};

export default nextConfig;
