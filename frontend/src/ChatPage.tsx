import {
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { API } from "./api";
import {
  CheckIcon,
  ChevronIcon,
  CrossIcon,
  PlusIcon,
  SendIcon,
  SparklesIcon,
  StopIcon,
  TrashIcon,
} from "./components/icons";
import { PageArt } from "./components/PageArt";
import "./ChatPage.css";

// PACo's agent, in PAC's process (GET /agent/status says whether it can run, and if not why).
interface AgentStatus {
  available: boolean;
  reason: string | null;
  model: string | null;
}

interface ChatEvent {
  index: number;
  kind: "user" | "step" | "answer" | "error" | "stopped";
  text: string;
}

interface EventsOut {
  events: ChatEvent[];
  busy: boolean;
  stopping?: boolean; // a stop was asked, and the answer has not ended yet
  progress: string | null;
  closed: boolean;
}

// A conversation the server keeps, as the list shows it.
interface SessionInfo {
  id: string;
  title: string; // its first question; empty before any
  busy: boolean;
  closed: boolean;
  started_at: string;
  updated_at: string;
}

// The conversation this tab shows, to find it again when the page comes back.
const SESSION_KEY = "pac.assistant.session";

function remembered(): string | null {
  try {
    return sessionStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

function remember(id: string | null) {
  try {
    if (id) sessionStorage.setItem(SESSION_KEY, id);
    else sessionStorage.removeItem(SESSION_KEY);
  } catch {
    // storage refused (a private window): the page just starts afresh
  }
}

/** "14:32", or "26 Sept" for another day. */
function when(iso: string): string {
  const date = new Date(iso);
  const today = new Date().toDateString() === date.toDateString();
  return today
    ? date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}

// A tool call the agent made. PACo reports it as "-> name(arguments)", followed by
// " refused: why" when PAC's host refused it; a line "failed: why" follows a call that failed.
interface Step {
  tool: string;
  args: Record<string, unknown>;
  state: "done" | "refused" | "failed";
  note: string | null;
}

// A question, the steps taken for it, and the answer, the error, or the stop.
interface Turn {
  key: number;
  question: string | null;
  steps: Step[];
  answer: string | null;
  error: string | null;
  stopped: string | null;
}

// A running tool's progress, e.g. "run_processing: 3 of 4 windows".
interface Progress {
  tool: string;
  text: string;
  fraction: number | null;
}

const EXAMPLES = [
  "Which profiles can I process?",
  "Process active_p1 and give me its dispersion curves.",
  "Process active_p1 and invert it: I want its velocity section.",
  "Process active_p1 and give me the soil types and the water table.",
];

const CALL = /^-> (\w+)\((.*)\)(?: refused: (.*))?$/s;

// Each tool's step when done, under way, and failed or refused; {name} stands for one of the
// call's arguments.
const LABELS: Record<string, [string, string, string]> = {
  list_profiles: [
    "Listed the profiles",
    "Listing the profiles",
    "List the profiles",
  ],
  inspect_profile: [
    "Inspected {profile}",
    "Inspecting {profile}",
    "Inspect {profile}",
  ],
  preset_settings: [
    "Read the processing settings",
    "Reading the processing settings",
    "Read the processing settings",
  ],
  run_processing: [
    "Processed {profile}",
    "Processing {profile}",
    "Process {profile}",
  ],
  pick: ["Picked the curves", "Picking the curves", "Pick the curves"],
  inversion_settings: [
    "Read the inversion settings",
    "Reading the inversion settings",
    "Read the inversion settings",
  ],
  invert: [
    "Started the inversion",
    "Starting the inversion",
    "Start the inversion",
  ],
  job_status: [
    "Followed the inversion",
    "Following the inversion",
    "Follow the inversion",
  ],
  petro_models: [
    "Compared the Silex models",
    "Comparing the Silex models",
    "Compare the Silex models",
  ],
  invert_petro: [
    "Ran the petrophysical inversion",
    "Running the petrophysical inversion",
    "Run the petrophysical inversion",
  ],
  redo: ["Redid the {stage}", "Redoing the {stage}", "Redo the {stage}"],
};

// PACo's host lists the parameters the stages ran with, then the settings the gates changed,
// after each answer, each under its title line.
const LISTS = ["Parameters used", "Settings the gates changed"];

async function detail(res: Response): Promise<string> {
  const body = await res.json().catch(() => null);
  return body?.detail ?? `HTTP ${res.status}`;
}

export default function ChatPage() {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [session, setSession] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [events, setEvents] = useState<ChatEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [closed, setClosed] = useState(false);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [unfolded, setUnfolded] = useState<ReadonlySet<number>>(new Set());
  const [listOpen, setListOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const pollRef = useRef<number | null>(null);
  const seenRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const checkStatus = useCallback(() => {
    fetch(`${API}/agent/status`)
      .then((res) => res.json())
      .then((data: AgentStatus) => setStatus(data))
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      );
  }, []);

  // The conversations the server keeps, the latest first.
  const refreshList = useCallback(async (): Promise<SessionInfo[]> => {
    try {
      const res = await fetch(`${API}/agent/sessions`);
      const listed = res.ok ? ((await res.json()) as SessionInfo[]) : [];
      setSessions(listed);
      return listed;
    } catch {
      return [];
    }
  }, []);

  // The conversations' menu closes on a click elsewhere, or Escape.
  useEffect(() => {
    if (!listOpen) return;
    function away(event: MouseEvent) {
      const target = event.target as Node;
      if (
        !menuRef.current?.contains(target) &&
        !toggleRef.current?.contains(target)
      )
        setListOpen(false);
    }
    function escape(event: KeyboardEvent) {
      if (event.key === "Escape") setListOpen(false);
    }
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", escape);
    };
  }, [listOpen]);

  // The newest message in view.
  useLayoutEffect(() => {
    const box = scrollRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [events, busy, progress]);

  // The text box grows with its text, up to its maximum height.
  useLayoutEffect(() => {
    const box = inputRef.current;
    if (!box) return;
    box.style.height = "auto";
    box.style.height = `${box.scrollHeight}px`;
  }, [text]);

  const poll = useCallback(
    (id: string) => {
      stopPolling();
      pollRef.current = window.setInterval(() => {
        fetch(`${API}/agent/sessions/${id}/events?after=${seenRef.current}`)
          .then(async (res) => {
            if (!res.ok) throw new Error(await detail(res));
            return (await res.json()) as EventsOut;
          })
          .then((body) => {
            if (body.events.length > 0) {
              seenRef.current += body.events.length;
              setEvents((previous) => [...previous, ...body.events]);
            }
            setBusy(body.busy);
            setStopping(body.stopping ?? false);
            setProgress(body.progress);
            setClosed(body.closed);
            if (!body.busy) {
              stopPolling();
              void refreshList();
            }
          })
          .catch((err) => {
            setError(err instanceof Error ? err.message : String(err));
            setBusy(false);
            stopPolling();
          });
      }, 1000);
    },
    [refreshList],
  );

  // A conversation shown whole, and followed while it answers.
  const open = useCallback(
    async (id: string) => {
      stopPolling();
      setError(null);
      const res = await fetch(`${API}/agent/sessions/${id}/events?after=0`);
      if (!res.ok) {
        remember(null);
        setSession(null);
        return;
      }
      const body = (await res.json()) as EventsOut;
      remember(id);
      setSession(id);
      setUnfolded(new Set());
      seenRef.current = body.events.length;
      setEvents(body.events);
      setBusy(body.busy);
      setStopping(body.stopping ?? false);
      setProgress(body.progress);
      setClosed(body.closed);
      if (body.busy) poll(id);
    },
    [poll],
  );

  // Back on the page: the conversation this tab showed, else one still answering.
  useEffect(() => {
    checkStatus();
    let cancelled = false;
    Promise.resolve()
      .then(refreshList)
      .then((listed) => {
        if (cancelled) return;
        const mine = remembered();
        const found =
          listed.find((one) => one.id === mine) ??
          listed.find((one) => one.busy);
        if (found) void open(found.id);
      });
    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [checkStatus, refreshList, open]);

  async function send() {
    const message = text.trim();
    if (!message || busy) return;
    setError(null);
    let id = session;
    if (!id || closed) {
      const res = await fetch(`${API}/agent/sessions`, { method: "POST" });
      if (!res.ok) {
        setError(await detail(res));
        return;
      }
      id = ((await res.json()) as { id: string }).id;
      remember(id);
      setSession(id);
      setEvents([]);
      setUnfolded(new Set());
      seenRef.current = 0;
      setClosed(false);
    }
    const res = await fetch(`${API}/agent/sessions/${id}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message }),
    });
    if (!res.ok) {
      setError(await detail(res));
      return;
    }
    setText("");
    setBusy(true);
    poll(id);
    void refreshList();
  }

  // The answer running, and all it started: at once, what had finished kept.
  function stop() {
    if (!session) return;
    setStopping(true);
    fetch(`${API}/agent/sessions/${session}/stop`, { method: "POST" }).catch(
      (err) => setError(err instanceof Error ? err.message : String(err)),
    );
  }

  // A conversation deleted: its answer stopped first if it runs.
  async function remove(id: string) {
    await fetch(`${API}/agent/sessions/${id}`, { method: "DELETE" }).catch(
      () => null,
    );
    if (id === session) newConversation();
    void refreshList();
  }

  // A new conversation; the others stay in the list.
  function newConversation() {
    stopPolling();
    remember(null);
    setSession(null);
    setEvents([]);
    setUnfolded(new Set());
    seenRef.current = 0;
    setBusy(false);
    setStopping(false);
    setProgress(null);
    setClosed(false);
    setError(null);
    inputRef.current?.focus();
  }

  function toggle(key: number) {
    setUnfolded((previous) => {
      const next = new Set(previous);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function start(example: string) {
    setText(example);
    inputRef.current?.focus();
  }

  const turns = turnsOf(events);
  const running = progressOf(progress);

  const listed = sessions.filter((one) => one.title || one.id === session);
  const answering = sessions.some((one) => one.busy);

  return (
    <div className="chat-page">
      <div className="chat-top">
        <header className="page-hero chat-hero">
          <div className="page-hero-text">
            <div className="page-icon">
              <SparklesIcon size={24} />
            </div>
            <div>
              <h1>AI assistant</h1>
              <p className="page-sub">
                Ask PACo to process, pick and invert your profiles
              </p>
            </div>
          </div>
          <div className="page-hero-art">
            <PageArt kind="assistant" />
          </div>
          <div className="chat-header-actions">
            <StatusBadge status={status} />
            <button
              ref={toggleRef}
              className={
                "chat-quiet chat-conversations" + (listOpen ? " open" : "")
              }
              onClick={() => {
                if (!listOpen) void refreshList();
                setListOpen(!listOpen);
              }}
              aria-expanded={listOpen}
              aria-haspopup="menu"
            >
              {answering && (
                <span className="chat-spinner" aria-label="One is answering" />
              )}
              Conversations
              <ChevronIcon size={14} />
            </button>
          </div>
        </header>

        {/* The conversations the server keeps, under their button: the header band clips. */}
        {listOpen && (
          <div
            className="chat-menu"
            ref={menuRef}
            role="menu"
            aria-label="Conversations"
          >
            <button
              className="chat-menu-new"
              role="menuitem"
              onClick={() => {
                newConversation();
                setListOpen(false);
              }}
              disabled={session === null}
            >
              <PlusIcon size={15} /> New conversation
            </button>
            {listed.length > 0 && (
              <div className="chat-menu-list">
                {listed.map((one) => (
                  <div
                    key={one.id}
                    className={
                      "chat-menu-row" + (one.id === session ? " active" : "")
                    }
                  >
                    <button
                      className="chat-menu-open"
                      role="menuitem"
                      onClick={() => {
                        void open(one.id);
                        setListOpen(false);
                      }}
                      aria-current={one.id === session ? "true" : undefined}
                    >
                      <span className="chat-menu-title">
                        {one.title || "New conversation"}
                      </span>
                      <span className="chat-menu-when">
                        {one.busy && (
                          <span
                            className="chat-spinner"
                            aria-label="Answering"
                          />
                        )}
                        {one.busy ? "answering" : when(one.updated_at)}
                      </span>
                    </button>
                    <button
                      className="chat-menu-delete"
                      onClick={() => void remove(one.id)}
                      aria-label="Delete"
                      data-tip={
                        "Delete\nA running answer is stopped first"
                      }
                    >
                      <TrashIcon size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <div
        className="chat-scroll"
        ref={scrollRef}
        role="log"
        aria-live="polite"
      >
        {status && !status.available && (
          <div className="chat-unavailable">
            <p style={{ margin: "0 0 6px", fontWeight: 600 }}>
              The AI assistant is not available.
            </p>
            <p style={{ margin: "0 0 6px" }}>{status.reason}</p>
            <p style={{ margin: "0 0 12px" }}>
              The rest of PAC works without it.
            </p>
            <button className="chat-quiet" onClick={checkStatus}>
              Check again
            </button>
          </div>
        )}
        {status?.available && turns.length === 0 && <Welcome onPick={start} />}
        {turns.map((turn, i) => (
          <TurnView
            key={turn.key}
            turn={turn}
            live={busy && i === turns.length - 1}
            progress={running}
            unfolded={unfolded.has(turn.key)}
            onToggle={() => toggle(turn.key)}
          />
        ))}
        {closed && (
          <p className="chat-note">
            This conversation has ended: your next message starts a new one.
          </p>
        )}
        {error && <div className="chat-error">{error}</div>}
      </div>

      {status?.available && (
        <div className="chat-composer-wrap">
          <div className="chat-composer">
            <textarea
              ref={inputRef}
              rows={1}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder="Ask PACo to process a profile, pick its curves or invert them…"
              aria-label="Your request"
            />
            {busy ? (
              <button
                className="chat-send chat-stop"
                onClick={stop}
                disabled={stopping}
                aria-label={stopping ? "Stopping" : "Stop"}
                data-tip={
                  stopping
                    ? "Stopping…"
                    : "Stop now\nThe answer and the runs it started\nWhat finished is kept"
                }
              >
                {stopping ? (
                  <span className="chat-spinner" />
                ) : (
                  <StopIcon size={16} />
                )}
              </button>
            ) : (
              <button
                className="chat-send"
                onClick={() => void send()}
                disabled={!text.trim()}
                aria-label="Send"
                title="Send"
              >
                <SendIcon size={18} />
              </button>
            )}
          </div>
          <p className="chat-hint">
            Enter to send, Shift+Enter for a new line.
          </p>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: AgentStatus | null }) {
  const ready = status?.available === true;
  const label =
    status === null
      ? "Checking…"
      : ready
        ? (status.model ?? "Ready")
        : "Unavailable";
  return (
    <span className="chat-status">
      <span className={ready ? "chat-status-dot ready" : "chat-status-dot"} />
      {label}
    </span>
  );
}

function Welcome({ onPick }: { onPick: (example: string) => void }) {
  return (
    <div className="chat-welcome">
      <div className="chat-welcome-icon">
        <SparklesIcon size={26} />
      </div>
      <h2 className="chat-welcome-title">What should PACo do?</h2>
      <p className="chat-welcome-text">
        Ask in plain words, or start from one of these.
      </p>
      <div className="chat-chips">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            className="chat-chip"
            onClick={() => onPick(example)}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}

function TurnView({
  turn,
  live,
  progress,
  unfolded,
  onToggle,
}: {
  turn: Turn;
  live: boolean;
  progress: Progress | null;
  unfolded: boolean;
  onToggle: () => void;
}) {
  const [answer, lists] =
    turn.answer === null ? ["", []] : splitLists(turn.answer);
  const count = turn.steps.length;
  const last = turn.steps[count - 1];
  // While the agent works, its latest call is under way; after a refused or failed one, it is
  // thinking about its next step.
  const underWay = live && last?.state === "done";
  const thinking = live && !underWay;
  return (
    <section className="chat-turn">
      {turn.question && (
        <div className="chat-question" data-kind="user">
          {turn.question}
        </div>
      )}
      {count > 0 && (
        <div>
          {!live && (
            <button
              className="chat-steps-toggle"
              aria-expanded={unfolded}
              onClick={onToggle}
            >
              <ChevronIcon size={14} />
              {count} {count === 1 ? "step" : "steps"}
            </button>
          )}
          {(live || unfolded) && (
            <ol className="chat-step-list">
              {turn.steps.map((step, i) => {
                const current = underWay && i === count - 1;
                // A tool that reports no progress must not show the previous tool's.
                const own =
                  current && progress?.tool === step.tool ? progress : null;
                return (
                  <StepRow
                    key={i}
                    step={step}
                    underWay={current}
                    progress={own}
                  />
                );
              })}
            </ol>
          )}
        </div>
      )}
      {thinking && (
        <div className="chat-thinking">
          <span className="chat-spinner" /> Thinking…
        </div>
      )}
      {turn.answer !== null && (
        <div className="chat-answer" data-kind="answer">
          <Markdown text={answer} />
          {lists.map((list) => (
            <div
              key={list.title}
              className={
                list.title === LISTS[0]
                  ? "chat-changes chat-used"
                  : "chat-changes"
              }
            >
              <div className="chat-changes-title">{list.title}</div>
              <ul>
                {list.items.map((item, i) => (
                  <li key={i}>{inline(item, `${list.title}-${i}`)}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
      {turn.error !== null && (
        <div className="chat-error" data-kind="error">
          {turn.error}
        </div>
      )}
      {turn.stopped !== null && (
        <p className="chat-stopped" data-kind="stopped">
          <StopIcon size={12} /> {turn.stopped}
        </p>
      )}
    </section>
  );
}

function StepRow({
  step,
  underWay,
  progress,
}: {
  step: Step;
  underWay: boolean;
  progress: Progress | null;
}) {
  const icon = underWay ? (
    <span className="chat-spinner" />
  ) : step.state === "done" ? (
    <CheckIcon size={15} />
  ) : (
    <CrossIcon size={15} />
  );
  return (
    <li
      className={`chat-step chat-step-${underWay ? "running" : step.state}`}
      data-kind="step"
    >
      <span className="chat-step-icon">{icon}</span>
      <span>
        {stepLabel(step, underWay)}
        {progress && (
          <span className="chat-step-progress"> · {progress.text}</span>
        )}
        {step.note && (
          <span className="chat-step-note">
            {step.state === "refused"
              ? `Not run: ${step.note}`
              : `Failed: ${step.note}`}
          </span>
        )}
        {progress?.fraction != null && (
          <span className="chat-bar">
            <span style={{ width: `${progress.fraction * 100}%` }} />
          </span>
        )}
      </span>
    </li>
  );
}

function turnsOf(events: ChatEvent[]): Turn[] {
  const turns: Turn[] = [];
  for (const event of events) {
    if (event.kind === "user" || turns.length === 0) {
      turns.push({
        key: event.index,
        question: event.kind === "user" ? event.text : null,
        steps: [],
        answer: null,
        error: null,
        stopped: null,
      });
      if (event.kind === "user") continue;
    }
    const turn = turns[turns.length - 1];
    if (event.kind === "step") {
      const call = CALL.exec(event.text);
      const last = turn.steps[turn.steps.length - 1];
      if (call) {
        turn.steps.push({
          tool: call[1],
          args: parseArgs(call[2]),
          state: call[3] ? "refused" : "done",
          note: call[3] ?? null,
        });
      } else if (event.text.startsWith("failed:") && last) {
        last.state = "failed";
        last.note = event.text.slice("failed:".length).trim();
      }
    } else if (event.kind === "answer") {
      turn.answer = event.text;
    } else if (event.kind === "stopped") {
      turn.stopped = event.text;
    } else {
      turn.error = event.text;
    }
  }
  return turns;
}

function parseArgs(text: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(text || "{}");
    return typeof parsed === "object" && parsed !== null
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

function stepLabel(step: Step, underWay: boolean): string {
  const labels = LABELS[step.tool];
  if (!labels) return step.tool;
  const form = underWay ? 1 : step.state === "done" ? 0 : 2;
  return labels[form].replace(/\{(\w+)\}/g, (_, name: string) => {
    const value = step.args[name];
    if (typeof value !== "string" || !value)
      return name === "profile" ? "a profile" : "stage";
    return name === "stage" ? value.replaceAll("_", " ") : value;
  });
}

function progressOf(progress: string | null): Progress | null {
  const found = progress ? /^(\w+): (.*)$/s.exec(progress) : null;
  if (!found) return null;
  const [, tool, text] = found;
  const count = /(\d+(?:\.\d+)?) of (\d+(?:\.\d+)?)/.exec(text);
  const fraction =
    count && Number(count[2]) > 0
      ? Math.min(1, Number(count[1]) / Number(count[2]))
      : null;
  return { tool, text, fraction };
}

interface AnswerList {
  title: string;
  items: string[];
}

// The answer, and the host's lists after it, in the order they come.
function splitLists(answer: string): [string, AnswerList[]] {
  const found = LISTS.map((title) => ({
    title,
    at: answer.lastIndexOf(`\n${title}:\n`),
  }))
    .filter((list) => list.at >= 0)
    .sort((a, b) => a.at - b.at);
  if (found.length === 0) return [answer, []];
  const lists = found.map(({ title, at }, i) => {
    const end = i + 1 < found.length ? found[i + 1].at : answer.length;
    const items = answer
      .slice(at + title.length + 3, end)
      .split("\n")
      .map((line) => line.replace(/^\s*[-*]\s+/, "").trim())
      .filter(Boolean);
    return { title, items };
  });
  return [answer.slice(0, found[0].at).trimEnd(), lists];
}

// The model writes Markdown: bold, code, headings and lists, drawn as elements (never as HTML).
function inline(text: string, key: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${key}-${i}`}>{part.slice(2, -2)}</strong>;
    }
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      return <code key={`${key}-${i}`}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let items: ReactNode[] = [];
  let ordered = false;
  const flush = (key: string) => {
    if (items.length === 0) return;
    blocks.push(
      ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>,
    );
    items = [];
  };
  text.split("\n").forEach((line, i) => {
    const bullet = /^(\s*)[-*]\s+(.*)$/.exec(line);
    const numbered = bullet ? null : /^(\s*)\d+[.)]\s+(.*)$/.exec(line);
    const item = bullet ?? numbered;
    if (item) {
      if (items.length > 0 && (numbered !== null) !== ordered)
        flush(`list-${i}`);
      ordered = numbered !== null;
      items.push(
        <li key={i} style={{ marginLeft: item[1].length * 6 }}>
          {inline(item[2], String(i))}
        </li>,
      );
      return;
    }
    flush(`list-${i}`);
    const heading = /^#{1,6}\s+(.*)$/.exec(line);
    if (heading) {
      blocks.push(
        <p key={i} className="chat-heading">
          {inline(heading[1], String(i))}
        </p>,
      );
    } else if (line.trim()) {
      blocks.push(<p key={i}>{inline(line, String(i))}</p>);
    }
  });
  flush("list-end");
  return <>{blocks}</>;
}
