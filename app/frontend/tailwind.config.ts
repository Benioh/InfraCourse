import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "quest-bg": "#f4f1eb",
        "quest-card": "#fffdf8",
        "quest-ink": "#1d1b18",
        "quest-muted": "#6d655d",
        "quest-border": "#d8cec1",
        "quest-accent": "#0f766e",
        "quest-warn": "#b45309",
        "quest-danger": "#b91c1c",
      },
      boxShadow: {
        panel: "0 20px 60px rgba(51, 45, 35, 0.08)",
      },
      borderRadius: {
        panel: "1.5rem",
      },
    },
  },
  plugins: [],
};

export default config;
