"use client";

import { useState } from "react";

export function CopyButton({ text, label = "复制" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="rounded-full border border-quest-border bg-white/90 px-3 py-1 text-xs font-medium text-quest-ink transition hover:border-quest-accent hover:text-quest-accent"
    >
      {copied ? "已复制" : label}
    </button>
  );
}
