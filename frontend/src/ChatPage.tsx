import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { API } from "./api";

// PACo's agent, in PAC's process (GET /agent/status says whether it can run, and if not why).
interface AgentStatus {
  available: boolean;
  reason: string | null;
  model: string | null;
}

interface ChatEvent {
  index: number;
  kind: "user" | "step" | "answer" | "error";
  text: string;
}

interface EventsOut {
  events: ChatEvent[];
  busy: boolean;
  progress: string | null;
  closed: boolean;
}

const EXAMPLES = [
  "Which profiles can I process?",
  "Process active_p1 and give me its dispersion curves.",
  "Process active_p1 and invert it: I want its velocity section.",
];

async function detail(res: Response): Promise<string> {
  const body = await res.json().catch(() => null);
  return body?.detail ?? `HTTP ${res.status}`;
}

export default function ChatPage() {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [session, setSession] = useState<string | null>(null);
  const [events, setEvents] = useState<ChatEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [closed, setClosed] = useState(false);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [showSteps, setShowSteps] = useState(true);
  const pollRef = useRef<number | null>(null);
  const seenRef = useRef(0);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => {
    fetch(`${API}/agent/status`)
      .then((res) => res.json())
      .then((data: AgentStatus) => setStatus(data))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
    return stopPolling;
  }, []);

  const poll = useCallback((id: string) => {
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
          setProgress(body.progress);
          setClosed(body.closed);
          if (!body.busy) stopPolling();
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : String(err));
          setBusy(false);
          stopPolling();
        });
    }, 1000);
  }, []);

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
      setSession(id);
      setEvents([]);
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
  }

  function newConversation() {
    stopPolling();
    if (session) fetch(`${API}/agent/sessions/${session}`, { method: "DELETE" }).catch(() => null);
    setSession(null);
    setEvents([]);
    seenRef.current = 0;
    setBusy(false);
    setProgress(null);
    setClosed(false);
    setError(null);
  }

  const shown = showSteps ? events : events.filter((event) => event.kind !== "step");

  return (
    <div style={{ padding: 24 }}>
      <h1>Assistant</h1>
      <p style={{ color: "var(--text-muted)" }}>
        Ask PACo, PAC's agent, to process a profile, pick its curves or invert them: it runs PAC's
        processing with quality gates, and its runs appear in the other pages.
      </p>

      {!status && !error && <p>Checking the assistant…</p>}
      {status && !status.available && (
        <div
          style={{
            background: "var(--info-bg)",
            color: "var(--info-text)",
            padding: "8px 12px",
            borderRadius: "var(--radius-sm)",
          }}
        >
          <p style={{ margin: "4px 0" }}>The assistant is not available: {status.reason}</p>
          <p style={{ margin: "4px 0" }}>The rest of PAC works without it.</p>
        </div>
      )}

      {status?.available && (
        <>
          <p style={{ color: "var(--text-faint)", fontSize: 13 }}>
            Model: {status.model}.{" "}
            <label style={{ marginLeft: 12 }}>
              <input type="checkbox" checked={showSteps} onChange={(e) => setShowSteps(e.target.checked)} />{" "}
              Show tool calls
            </label>
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: 8, margin: "12px 0" }}>
            {shown.map((event) => (
              <Message key={event.index} event={event} />
            ))}
            {busy && (
              <p style={{ color: "var(--text-muted)", fontStyle: "italic", margin: 0 }}>
                Working…{progress ? ` ${progress}` : ""}
              </p>
            )}
          </div>

          {events.length === 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 8 }}>
              {EXAMPLES.map((example) => (
                <button key={example} onClick={() => setText(example)} style={{ fontSize: 13 }}>
                  {example}
                </button>
              ))}
            </div>
          )}

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder="Your request (Enter to send, Shift+Enter for a new line)"
            rows={3}
            style={{ width: "100%", boxSizing: "border-box" }}
            disabled={busy}
          />
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <button onClick={() => void send()} disabled={busy || !text.trim()}>
              {busy ? "Working…" : "Send"}
            </button>
            <button onClick={newConversation} disabled={busy}>
              New conversation
            </button>
          </div>
          {closed && (
            <p style={{ color: "var(--text-muted)" }}>
              This conversation has ended; your next message starts a new one.
            </p>
          )}
        </>
      )}

      {error && <p style={{ color: "crimson" }}>{error}</p>}
    </div>
  );
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
  const flush = (key: string) => {
    if (items.length > 0) {
      blocks.push(<ul key={key} style={{ margin: "4px 0", paddingLeft: 20 }}>{items}</ul>);
      items = [];
    }
  };
  text.split("\n").forEach((line, i) => {
    const bullet = line.match(/^(\s*)[-*]\s+(.*)$/);
    if (bullet) {
      items.push(
        <li key={i} style={{ marginLeft: bullet[1].length * 6 }}>
          {inline(bullet[2], String(i))}
        </li>,
      );
      return;
    }
    flush(`list-${i}`);
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (heading) {
      blocks.push(<p key={i} style={{ margin: "6px 0", fontWeight: 600 }}>{inline(heading[1], String(i))}</p>);
    } else if (line.trim()) {
      blocks.push(<p key={i} style={{ margin: "2px 0" }}>{inline(line, String(i))}</p>);
    }
  });
  flush("list-end");
  return <>{blocks}</>;
}

function Message({ event }: { event: ChatEvent }) {
  if (event.kind === "step") {
    return (
      <code
        data-kind="step"
        style={{ fontSize: 12, color: "var(--text-faint)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}
      >
        {event.text}
      </code>
    );
  }
  const user = event.kind === "user";
  return (
    <div
      data-kind={event.kind}
      style={{
        alignSelf: user ? "flex-end" : "flex-start",
        maxWidth: "85%",
        background: user ? "var(--accent-soft)" : "var(--surface)",
        color: event.kind === "error" ? "crimson" : "var(--text)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-sm)",
        padding: "8px 12px",
        whiteSpace: event.kind === "answer" ? "normal" : "pre-wrap",
        wordBreak: "break-word",
      }}
    >
      {event.kind === "answer" ? <Markdown text={event.text} /> : event.text}
    </div>
  );
}
