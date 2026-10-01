import { useEffect, useState } from "react";
import type { Conversation } from "./running";
import { readStored, writeStored } from "./stored";

// What the assistant said while its page was not open, for the side menu: a new answer, or
// one that asks the user to choose among the options a tool offered. Opening the page sees
// them. What was seen is kept in this tab's session storage: a reload keeps it, and a new tab
// takes what is there as seen.

export type News = "asks" | "answer" | null;

const KEY = "assistant.seen";

/** The answers seen, by conversation. */
type Seen = Record<string, number>;

function counted(conversations: readonly Conversation[]): Seen {
  return Object.fromEntries(conversations.map((one) => [one.id, one.answers]));
}

/** The assistant's news since its page was last open (none while it is open, or before the
 * conversations are first heard of). */
export function useAssistantNews(conversations: readonly Conversation[] | null, onPage: boolean): News {
  const [seen, setSeen] = useState<Seen | undefined>(() => readStored<Seen>(KEY));
  // The page open, or the first answers this tab hears of: all seen.
  if (conversations !== null && (onPage || seen === undefined)) {
    const now = counted(conversations);
    if (seen === undefined || !sameCounts(seen, now)) setSeen(now);
  }

  useEffect(() => {
    if (seen !== undefined) writeStored(KEY, seen);
  }, [seen]);

  if (conversations === null || onPage || seen === undefined) return null;
  const fresh = conversations.filter((one) => !one.busy && one.answers > (seen[one.id] ?? 0));
  if (fresh.length === 0) return null;
  return fresh.some((one) => one.asks) ? "asks" : "answer";
}

function sameCounts(a: Seen, b: Seen): boolean {
  const keys = Object.keys(b);
  return keys.length === Object.keys(a).length && keys.every((key) => a[key] === b[key]);
}
