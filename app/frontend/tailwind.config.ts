import type { Config } from "tailwindcss";

// Morandi-inspired palette: low-saturation neutrals + deep sage accent.
// Tokens are intentionally narrow — every UI surface picks from this set.
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
    "./app/frontend/app/**/*.{ts,tsx}",
    "./app/frontend/components/**/*.{ts,tsx}",
    "./app/frontend/lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Surfaces
        "quest-bg": "#f1f0ec",          // page — warm mist white
        "quest-card": "#fbfaf7",         // card — soft ivory
        "quest-card-soft": "#f6f4ef",    // sub-card / hover row

        // Ink
        "quest-ink": "#2c2e36",          // primary text — cool slate
        "quest-ink-soft": "#4a4d57",     // secondary heading
        "quest-muted": "#8a8c92",        // metadata / placeholder

        // Edges
        "quest-border": "#e4e2dc",       // default fine border
        "quest-border-soft": "#ece9e2",  // hairline / divider

        // Accents — pick from a tight Morandi set, never mix more than two on one surface
        "quest-accent": "#6b8073",       // sage (primary action, link)
        "quest-accent-soft": "#dfe5e0",  // sage tint (hover, chip bg)
        "quest-mist": "#7d96a4",         // fog blue (secondary)
        "quest-mist-soft": "#dde4ea",    // fog blue tint
        "quest-dust": "#a8736f",         // dust rose (pending, attention)
        "quest-dust-soft": "#ede0dd",    // dust rose tint
        "quest-sand": "#d4c8b6",         // warm sand (highlight bg)
        "quest-sand-soft": "#ece5d6",    // sand tint (line-hit)

        // Status (mapped to Morandi tones, not bright Tailwind defaults)
        "quest-warn": "#a08056",         // muted ochre
        "quest-danger": "#9a6b6b",       // muted terracotta
        "quest-success": "#6b8073",      // alias of accent for clarity
      },
      boxShadow: {
        panel: "0 8px 28px rgba(60, 60, 67, 0.06)",
        "panel-soft": "0 1px 3px rgba(60, 60, 67, 0.04)",
        drawer: "-12px 0 36px rgba(60, 60, 67, 0.08)",
      },
      borderRadius: {
        panel: "1.25rem",
        chip: "0.625rem",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      letterSpacing: {
        eyebrow: "0.22em",
      },
    },
  },
  plugins: [],
};

export default config;
