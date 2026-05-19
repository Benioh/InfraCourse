"use client";

import { useCallback } from "react";

export type SelectionContext = {
  text: string;
  sourcePath: string | null;
  lines: [number, number] | null;
};

const EMPTY: SelectionContext = { text: "", sourcePath: null, lines: null };

export function useSelection(): () => SelectionContext {
  return useCallback(() => readSelection(), []);
}

export function readSelection(): SelectionContext {
  if (typeof window === "undefined") return EMPTY;
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return EMPTY;
  const text = sel.toString();
  if (!text.trim()) return EMPTY;

  const range = sel.getRangeAt(0);
  const startNode = range.startContainer;
  const endNode = range.endContainer;

  const sourcePath = findAncestorAttr(startNode, "data-source-path");

  // Walk up from each endpoint to find the .line element (Shiki tags each
  // rendered line with `data-line` in source/page.tsx via lineDecorators).
  const startLine = findLineNumber(startNode);
  const endLine = findLineNumber(endNode);

  let lines: [number, number] | null = null;
  if (startLine != null && endLine != null) {
    lines = startLine <= endLine ? [startLine, endLine] : [endLine, startLine];
  } else if (startLine != null) {
    lines = [startLine, startLine];
  } else if (endLine != null) {
    lines = [endLine, endLine];
  }

  return { text, sourcePath, lines };
}

function findAncestorAttr(node: Node | null, attr: string): string | null {
  let cur: Node | null = node;
  while (cur) {
    if (cur.nodeType === Node.ELEMENT_NODE) {
      const value = (cur as Element).getAttribute(attr);
      if (value) return value;
    }
    cur = cur.parentNode;
  }
  return null;
}

function findLineNumber(node: Node | null): number | null {
  const raw = findAncestorAttr(node, "data-line");
  if (!raw) return null;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}
