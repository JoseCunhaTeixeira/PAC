import { createContext, useContext } from "react";
import type { Maker } from "./types";

// Who made the run the page shows: on a workflow run, whose request the assistant writes, a
// setting's "You" may be the assistant's choice.
export const RunMaker = createContext<Maker | null>(null);

export function useRunMaker(): Maker | null {
  return useContext(RunMaker);
}
