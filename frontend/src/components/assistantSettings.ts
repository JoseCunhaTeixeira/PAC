import { createContext, useContext } from "react";

// The assistant's settings (GET /agent/settings), as the settings popup shows them: only what is
// set (saved by the popup, or in .env), nothing made up. The API key is never sent back: whether
// one is set.
export interface AssistantSettings {
  mode: "autonomous" | "workflow" | null;
  provider: string | null; // a provider's ("anthropic", "openai", "gemini", "mistral"), or "server"
  model: string;
  base_url: string;
  context: number | null;
  key_set: boolean;
}

/** Whether the settings let the assistant run: a mode, a model, and where it is (a server's
 * address, a provider's key). */
export function complete(settings: AssistantSettings): boolean {
  if (!settings.mode || !settings.provider || !settings.model) return false;
  return settings.provider === "server" ? settings.base_url !== "" : settings.key_set;
}

// The settings popup, held by the app: `open` shows it over any page; `saved` counts its saves,
// for a page to read the settings again.
export interface SettingsPopup {
  open: () => void;
  saved: number;
}

export const SettingsPopupContext = createContext<SettingsPopup>({ open: () => {}, saved: 0 });

export function useSettingsPopup(): SettingsPopup {
  return useContext(SettingsPopupContext);
}
