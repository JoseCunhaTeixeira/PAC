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
// plotting components look these up explicitly via useTheme(): the page's own
// text colours (index.css), ticks and labels muted, titles in the main text's.
// A section's depths the data do not inform: veiled in the card's surface
// (--surface), below a line in Visualization's warn amber.
export function canvasPalette(theme: Theme) {
  return theme === "dark"
    ? {
        axis: "#5c6778",
        tick: "#a1abbc",
        title: "#e7ebf2",
        veil: "rgba(17, 23, 31, 0.62)",
        informed: "#f0b429",
      }
    : {
        axis: "#a3abb8",
        tick: "#525d70",
        title: "#0f1728",
        veil: "rgba(255, 255, 255, 0.62)",
        informed: "#d98a04",
      };
}

// The page's font (index.css), so that a plot's text reads as the page's.
export const FONT_FAMILY =
  '"Inter", "Source Sans Pro", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

/** The canvas font at `size` px (and `weight`), in the page's family. */
export function canvasFont(size: number, weight = 400): string {
  return `${weight} ${size}px ${FONT_FAMILY}`;
}

// Every plot's text (ticks, axes, colour bars, lanes) at one size: the pages' small text
// (index.css's --fs-small, 0.82rem of 15px).
export const CANVAS_FONT = canvasFont(12);
