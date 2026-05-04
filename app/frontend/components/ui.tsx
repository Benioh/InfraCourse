import Link from "next/link";
import type { ReactNode } from "react";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-quest-bg text-quest-ink">
      <div className="mx-auto max-w-7xl px-6 py-8">
        <header className="mb-8 overflow-hidden rounded-panel border border-quest-border bg-quest-card shadow-panel">
          <div className="bg-[radial-gradient(circle_at_top_left,_rgba(15,118,110,0.15),_transparent_45%),linear-gradient(135deg,_rgba(176,138,87,0.12),_transparent_55%)] px-6 py-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.3em] text-quest-muted">
                  任务控制台
                </p>
                <h1 className="mt-2 font-mono text-3xl font-semibold">Infra Quest</h1>
                <p className="mt-2 max-w-3xl text-sm text-quest-muted">
                  多模态大模型 Infra 工程训练系统：构建、破坏、调试并扩展训练/推理/RL 栈。
                </p>
              </div>
              <nav className="flex flex-wrap gap-2 text-sm">
                <NavLink href="/">总览</NavLink>
                <NavLink href="/curriculum">自学路线</NavLink>
                <NavLink href="/concept-map">概念图</NavLink>
                <NavLink href="/source-map/Megatron">源码地图</NavLink>
                <NavLink href="/source">源码浏览</NavLink>
                <NavLink href="/tickets">Debug 工单</NavLink>
                <NavLink href="/metrics">指标</NavLink>
                <NavLink href="/prompts">提示卡</NavLink>
                <NavLink href="/badges">徽章</NavLink>
              </nav>
            </div>
          </div>
        </header>
        {children}
      </div>
    </div>
  );
}

function NavLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link
      href={href}
      className="rounded-full border border-quest-border bg-white/80 px-4 py-2 font-medium text-quest-ink transition hover:border-quest-accent hover:text-quest-accent"
    >
      {children}
    </Link>
  );
}

export function Panel({
  title,
  eyebrow,
  children,
}: {
  title?: string;
  eyebrow?: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-panel border border-quest-border bg-quest-card p-5 shadow-panel">
      {eyebrow ? <p className="text-xs font-semibold uppercase tracking-[0.24em] text-quest-muted">{eyebrow}</p> : null}
      {title ? <h2 className="mt-2 font-mono text-xl font-semibold">{title}</h2> : null}
      <div className={title || eyebrow ? "mt-4" : ""}>{children}</div>
    </section>
  );
}

export function StatusPill({ passed, label }: { passed: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${
        passed ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
      }`}
    >
      {passed ? "已通过" : "待完成"} · {label}
    </span>
  );
}

export function CommandBlock({ command }: { command: string }) {
  return (
    <pre className="overflow-x-auto rounded-2xl border border-quest-border bg-[#201d18] p-4 text-sm text-[#f5ecdf]">
      <code>{command}</code>
    </pre>
  );
}

export function KeyValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-2xl border border-quest-border/70 bg-white/70 p-4">
      <div className="text-xs uppercase tracking-[0.22em] text-quest-muted">{label}</div>
      <div className="mt-2 text-lg font-semibold">{value}</div>
    </div>
  );
}
