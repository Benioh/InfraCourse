import Link from "next/link";
import type { ReactNode } from "react";
import { withPublicPrefix } from "@/lib/runtime";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-quest-bg text-quest-ink">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <header className="mb-10 overflow-hidden rounded-panel border border-quest-border bg-quest-card shadow-panel-soft">
          <div className="bg-[radial-gradient(circle_at_top_left,_rgba(107,128,115,0.08),_transparent_55%),linear-gradient(135deg,_rgba(125,150,164,0.06),_transparent_60%)] px-8 py-9">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <p className="text-[11px] font-medium uppercase tracking-eyebrow text-quest-muted">
                  任务控制台
                </p>
                <h1 className="mt-3 font-mono text-[2rem] font-semibold tracking-tight text-quest-ink">
                  Infra Quest
                </h1>
                <p className="mt-3 max-w-3xl text-sm leading-relaxed text-quest-ink-soft">
                  多模态大模型 Infra 工程训练系统：构建、破坏、调试并扩展训练 / 推理 / RL 栈。
                </p>
              </div>
              <nav className="flex flex-wrap gap-1.5 text-[13px]">
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
      href={withPublicPrefix(href)}
      className="rounded-chip border border-quest-border bg-quest-card px-3.5 py-1.5 font-medium text-quest-ink-soft transition-colors duration-150 hover:border-quest-accent hover:bg-quest-accent-soft hover:text-quest-accent"
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
    <section className="rounded-panel border border-quest-border bg-quest-card p-7 shadow-panel-soft">
      {eyebrow ? (
        <p className="text-[11px] font-medium uppercase tracking-eyebrow text-quest-muted">
          {eyebrow}
        </p>
      ) : null}
      {title ? (
        <h2 className="mt-2 font-mono text-xl font-semibold tracking-tight text-quest-ink">
          {title}
        </h2>
      ) : null}
      <div className={title || eyebrow ? "mt-5" : ""}>{children}</div>
    </section>
  );
}

export function StatusPill({ passed, label }: { passed: boolean; label: string }) {
  const tone = passed
    ? "bg-quest-accent-soft text-quest-accent"
    : "bg-quest-dust-soft text-quest-dust";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-medium tracking-wide ${tone}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${passed ? "bg-quest-accent" : "bg-quest-dust"}`}
        aria-hidden
      />
      {passed ? "已通过" : "待完成"} · {label}
    </span>
  );
}

export function CommandBlock({ command }: { command: string }) {
  // Code blocks stay dark for terminal feel, but cooled-down to match palette.
  return (
    <pre className="overflow-x-auto rounded-2xl border border-quest-border bg-[#23252b] p-4 text-[13px] leading-relaxed text-[#e8e9ec]">
      <code className="font-mono">{command}</code>
    </pre>
  );
}

export function KeyValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-2xl border border-quest-border bg-quest-card-soft p-5 transition-colors duration-150 hover:border-quest-accent/50">
      <div className="text-[11px] font-medium uppercase tracking-eyebrow text-quest-muted">
        {label}
      </div>
      <div className="mt-2 text-lg font-semibold text-quest-ink">{value}</div>
    </div>
  );
}
