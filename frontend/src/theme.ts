import { createContext, useContext } from "react";

export type Theme = "light" | "dark";

export function getInitialTheme(): Theme {
  const stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(theme: Theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
}

export const ThemeContext = createContext<Theme>("light");

export function useTheme(): Theme {
  return useContext(ThemeContext);
}

// Canvas drawing uses raw pixel colors that can't follow CSS variables, so
// plotting components look these up explicitly via useTheme().
export function canvasPalette(theme: Theme) {
  return theme === "dark"
    ? { axis: "#777", tick: "#ccc", title: "#f0f0f0" }
    : { axis: "#999", tick: "#444", title: "#222" };
}

// Shared so every canvas-based plot (dispersion image, pseudo-section, mute
// gather) renders axis/tick/title text at the same size.
export const CANVAS_FONT = "13px sans-serif";
