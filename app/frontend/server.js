const { createServer, request } = require("node:http");
const next = require("next");

process.chdir(__dirname);

function optionValue(names, fallback) {
  for (let i = 2; i < process.argv.length; i += 1) {
    const arg = process.argv[i];
    for (const name of names) {
      if (arg === name && process.argv[i + 1]) return process.argv[i + 1];
      if (arg.startsWith(`${name}=`)) return arg.slice(name.length + 1);
    }
  }
  return fallback;
}

function normalizePrefix(value) {
  if (!value) return "";
  const trimmed = value.trim().replace(/\/+$/, "");
  if (!trimmed) return "";
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

function deriveProxyPrefix(port) {
  if (process.env.NEXT_PUBLIC_PROXY_PREFIX !== undefined) {
    return normalizePrefix(process.env.NEXT_PUBLIC_PROXY_PREFIX);
  }
  const template = process.env.VSCODE_PROXY_URI;
  if (!template) return "";
  try {
    const url = new URL(template.replace("{{port}}", String(port)));
    return normalizePrefix(url.pathname);
  } catch {
    return "";
  }
}

const port = Number.parseInt(optionValue(["--port", "-p"], process.env.PORT ?? "3000"), 10);
const hostname = optionValue(["--hostname", "-H"], process.env.HOST ?? "0.0.0.0");
const dev = process.env.NODE_ENV !== "production";
const proxyPrefix = deriveProxyPrefix(port);

process.env.PORT = String(port);
if (proxyPrefix) {
  process.env.NEXT_PUBLIC_PROXY_PREFIX = proxyPrefix;
}

const app = next({ dev, hostname, port, dir: __dirname });
const handle = app.getRequestHandler();

function proxyStrippedNextAsset(req, res) {
  const upstream = request(
    {
      hostname: "127.0.0.1",
      port,
      method: req.method,
      path: `${proxyPrefix}${req.url}`,
      headers: req.headers,
    },
    (upstreamRes) => {
      res.writeHead(upstreamRes.statusCode ?? 502, upstreamRes.headers);
      upstreamRes.pipe(res);
    },
  );
  upstream.on("error", (err) => {
    res.writeHead(502, { "content-type": "text/plain; charset=utf-8" });
    res.end(`Failed to proxy Next asset: ${err.message}`);
  });
  req.pipe(upstream);
}

app.prepare().then(() => {
  createServer((req, res) => {
    if (proxyPrefix && req.url && req.url.startsWith("/_next/")) {
      proxyStrippedNextAsset(req, res);
      return;
    }
    if (proxyPrefix && req.url && (req.url === proxyPrefix || req.url.startsWith(`${proxyPrefix}/`))) {
      const withoutPrefix = req.url.slice(proxyPrefix.length) || "/";
      if (!withoutPrefix.startsWith("/_next/")) {
        req.url = withoutPrefix;
      }
    }
    handle(req, res);
  }).listen(port, hostname, () => {
    const local = `http://localhost:${port}`;
    console.log(`> Ready on ${local}`);
    if (proxyPrefix) {
      console.log(`> VS Code proxy prefix: ${proxyPrefix}`);
    }
  });
});
