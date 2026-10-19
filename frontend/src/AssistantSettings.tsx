import { useEffect, useRef, useState, type ReactNode } from "react";
import { API } from "./api";
import { Callout, Segmented, type SegmentOption } from "./components/kit";
import { CrossIcon, LockIcon } from "./components/icons";
import type { AssistantSettings } from "./components/assistantSettings";
import "./AssistantSettings.css";

// The assistant's settings, in a popup over any page (GET/PUT /agent/settings): the mode, the
// model and whether figures may reach it. Nothing is chosen for the user: the popup starts from
// what is set (saved here, or in .env), and saves once a mode and a model are chosen.

type Mode = "autonomous" | "workflow";
// Where the model is: a provider's API, or the user's server.
type Way = "api" | "server";
type Api = "anthropic" | "openai" | "gemini" | "mistral";

interface Choice<T extends string> {
  value: T;
  label: string;
  note?: string;
}

const MODES: Choice<Mode>[] = [
  { value: "autonomous", label: "Autonomous", note: "The model works on its own. It needs a frontier model." },
  { value: "workflow", label: "Workflow", note: "PACo guides the model step by step. A small model on your GPU is enough." },
];

const WAYS: Choice<Way>[] = [
  { value: "api", label: "A provider's API", note: "Claude, OpenAI, Gemini or Mistral, with your key" },
  { value: "server", label: "Your server", note: "vLLM, Ollama, LM Studio, OpenRouter or a gateway" },
];

const APIS: Choice<Api>[] = [
  { value: "anthropic", label: "Claude" },
  { value: "openai", label: "OpenAI" },
  { value: "gemini", label: "Gemini" },
  { value: "mistral", label: "Mistral" },
];

const VISION: SegmentOption<"off" | "on">[] = [
  { value: "off", label: "Off" },
  { value: "on", label: "On" },
];

// A connection test's limit, in seconds, as the backend's (TEST_S): the button counts it down.
const TEST_S = 60;
// The context the autonomous mode needs, and the one a server is taken to serve when none is set.
const LONG_CONTEXT = 131_072;
const SERVER_CONTEXT = 16_384;

interface Form {
  mode: Mode | null;
  way: Way | null;
  api: Api | null;
  model: string;
  baseUrl: string;
  context: string; // as typed; empty: none
  vision: boolean;
}

const EMPTY: Form = { mode: null, way: null, api: null, model: "", baseUrl: "", context: "", vision: true };

// A connection test (POST /agent/settings/test): one short call, never a figure.
interface Tested {
  ok: boolean;
  seconds: number | null;
  images: boolean | null;
  error: string | null;
}

function formOf(settings: AssistantSettings): Form {
  const api = APIS.find((one) => one.value === settings.provider)?.value ?? null;
  return {
    mode: settings.mode,
    way: settings.provider === "server" ? "server" : api ? "api" : null,
    api,
    model: settings.model,
    baseUrl: settings.base_url,
    context: settings.context === null ? "" : String(settings.context),
    vision: settings.vision,
  };
}

function providerOf(way: Way | null, api: Api | null): string | null {
  return way === "server" ? "server" : way === "api" ? api : null;
}


function contextOf(form: Form): number | null {
  return form.context.trim() === "" ? null : Number(form.context);
}

// The settings as the backend takes them, the key apart.
function bodyOf(form: Form) {
  const atServer = form.way === "server";
  return {
    mode: form.mode,
    provider: providerOf(form.way, form.api),
    model: form.model.trim(),
    base_url: atServer ? form.baseUrl.trim() : "",
    context: atServer ? contextOf(form) : null,
    vision: form.mode === "autonomous" && form.vision,
  };
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

async function detail(res: Response): Promise<string> {
  const body = await res.json().catch(() => null);
  return typeof body?.detail === "string" ? body.detail : `HTTP ${res.status}`;
}

/** The settings popup: shown while `open`, a fresh form each time; locked while `answering`. */
export default function AssistantSettingsPopup({
  open,
  answering,
  onClose,
  onSaved,
}: {
  open: boolean;
  answering: boolean;
  onClose: () => void;
  onSaved: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    else if (!open && dialog.open) dialog.close();
  }, [open]);
  return (
    <dialog ref={ref} className="settings-dialog" aria-labelledby="settings-title" onClose={onClose}>
      {open && <SettingsForm answering={answering} onClose={onClose} onSaved={onSaved} />}
    </dialog>
  );
}

function SettingsForm({
  answering,
  onClose,
  onSaved,
}: {
  answering: boolean;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [saved, setSaved] = useState<AssistantSettings | null>(null);
  const [form, setForm] = useState<Form>(EMPTY);
  // A key typed; empty: the one set kept (for the same provider), else none.
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [left, setLeft] = useState(TEST_S);
  const [tested, setTested] = useState<Tested | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/agent/settings`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await detail(res));
        return (await res.json()) as AssistantSettings;
      })
      .then((found) => {
        if (cancelled) return;
        setSaved(found);
        setForm(formOf(found));
      })
      .catch((err) => {
        if (!cancelled) setError(message(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // The test's countdown, a second at a time while it runs.
  useEffect(() => {
    if (!testing) return;
    const timer = window.setInterval(() => setLeft((seconds) => Math.max(0, seconds - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [testing]);

  const provider = providerOf(form.way, form.api);
  const atServer = form.way === "server";
  const sameProvider = saved !== null && provider !== null && provider === saved.provider;
  const keyKept = sameProvider && saved.key_set;
  const context = contextOf(form);
  const contextOk = context === null || (Number.isInteger(context) && context > 0);
  // Enough to reach the model: where it is, its name, a key at a provider.
  const reachable =
    provider !== null &&
    form.model.trim() !== "" &&
    (!atServer || form.baseUrl.trim() !== "") &&
    (atServer || key.trim() !== "" || keyKept) &&
    contextOk;
  const short = form.mode === "autonomous" && atServer && contextOk && (context ?? SERVER_CONTEXT) < LONG_CONTEXT;

  // The settings as the backend takes them: the key typed, else the one set (null) for the same
  // provider, never another provider's ("").
  function body() {
    return { ...bodyOf(form), api_key: key.trim() !== "" ? key.trim() : sameProvider ? null : "" };
  }
  // The settings in force, unchanged: nothing to save. Saved changes clear the conversations, so
  // never while an answer runs.
  const unchanged =
    saved !== null && key.trim() === "" && JSON.stringify(bodyOf(form)) === JSON.stringify(bodyOf(formOf(saved)));
  const current = unchanged && form.mode !== null && reachable;
  const canSave = form.mode !== null && reachable && !unchanged && !answering && !saving;

  function change(patch: Partial<Form>) {
    setForm((previous) => ({ ...previous, ...patch }));
    setTested(null);
    setError(null);
  }

  // Another model's place: its own settings again if it is the one saved, else none.
  function choose(way: Way, api: Api | null) {
    const own = saved !== null && providerOf(way, api) === saved.provider ? formOf(saved) : null;
    change({ way, api, model: own?.model ?? "", baseUrl: own?.baseUrl ?? "", context: own?.context ?? "" });
    setKey("");
  }

  async function test() {
    setTested(null);
    setLeft(TEST_S);
    setTesting(true);
    try {
      const res = await fetch(`${API}/agent/settings/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body()),
      });
      if (!res.ok) throw new Error(await detail(res));
      setTested((await res.json()) as Tested);
    } catch (err) {
      setTested({ ok: false, seconds: null, images: null, error: message(err) });
    } finally {
      setTesting(false);
    }
  }

  async function save() {
    if (!canSave) return;
    setSaving(true);
    setError(null);
    try {
      // Vision kept as chosen in either mode: off stays off when the workflow mode is saved.
      const res = await fetch(`${API}/agent/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body(), vision: form.vision }),
      });
      if (!res.ok) throw new Error(await detail(res));
    } catch (err) {
      setError(message(err));
      setSaving(false);
      return;
    }
    onSaved();
    onClose();
  }

  return (
    <form
      className="settings-form"
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
    >
      <header className="settings-head">
        <div>
          <h2 id="settings-title">Assistant settings</h2>
          <p>How the AI assistant works, and its model.</p>
        </div>
        <button type="button" className="ghost settings-close" aria-label="Close" onClick={onClose}>
          <CrossIcon size={16} />
        </button>
      </header>

      <div className="settings-body">
        {saved === null && !error && <p className="settings-note">Loading…</p>}
        {saved !== null && (
          <fieldset className="settings-sections" disabled={answering}>
            <section>
              <h3>Mode</h3>
              <Choices name="mode" label="Mode" options={MODES} value={form.mode} onChange={(mode) => change({ mode })} />
            </section>

            {form.mode === "autonomous" && (
              <section>
                <h3>Vision</h3>
                <div className="settings-switch">
                  <Segmented
                    label="Vision"
                    value={form.vision ? "on" : "off"}
                    options={VISION}
                    onChange={(value) => change({ vision: value === "on" })}
                  />
                  <span className="settings-note">
                    {form.vision ? "Figures go to the model's provider." : "Figures stay on this machine."}
                  </span>
                </div>
              </section>
            )}

            <section>
              <h3>Model</h3>
              <Choices
                name="way"
                label="Where the model is"
                options={WAYS}
                value={form.way}
                onChange={(way) => choose(way, way === "api" && saved !== null ? formOf(saved).api : null)}
              />
              {form.way === "api" && (
                <Choices name="api" label="Provider" options={APIS} value={form.api} onChange={(api) => choose("api", api)} compact />
              )}
              {provider !== null && (
                <div className="settings-fields">
                  {atServer && (
                    <Field label="Address">
                      <input
                        type="url"
                        value={form.baseUrl}
                        placeholder="e.g. http://gpu-host:8001/v1"
                        spellCheck={false}
                        autoComplete="off"
                        onChange={(e) => change({ baseUrl: e.target.value })}
                      />
                    </Field>
                  )}
                  <Field label="Model">
                    <input
                      type="text"
                      value={form.model}
                      placeholder="The model's name"
                      spellCheck={false}
                      autoComplete="off"
                      onChange={(e) => change({ model: e.target.value })}
                    />
                  </Field>
                  <Field
                    label={
                      <>
                        <LockIcon size={13} /> API key{atServer ? " (optional)" : ""}
                      </>
                    }
                    hint="Kept on this machine, never shown again"
                  >
                    <input
                      type="password"
                      value={key}
                      placeholder={keyKept ? "••••••••••••••••" : atServer ? "If the server asks for one" : "Your key"}
                      autoComplete="new-password"
                      spellCheck={false}
                      onChange={(e) => {
                        setKey(e.target.value);
                        setTested(null);
                      }}
                    />
                  </Field>
                  {atServer && (
                    <Field label="Context (optional)" hint="Empty: 16,384">
                      <input
                        type="number"
                        min={1}
                        value={form.context}
                        className={contextOk ? "" : "invalid"}
                        onChange={(e) => change({ context: e.target.value })}
                      />
                      <span className="field-unit">tokens</span>
                    </Field>
                  )}
                </div>
              )}
              {short && (
                <Callout tone="warn" title="Context too short for the autonomous mode">
                  Give it 131,072 tokens or more, or choose a provider&apos;s API.
                </Callout>
              )}
            </section>
          </fieldset>
        )}
        {answering && (
          <Callout tone="warn" title="An answer is running">
            Let it finish, or stop it, to change the settings. Saving new settings clears the conversations.
          </Callout>
        )}
        {error && <Callout tone="error">{error}</Callout>}
      </div>

      <footer className="settings-foot">
        <div className="settings-test">
          <button type="button" className="secondary" onClick={() => void test()} disabled={testing || !reachable}>
            {testing ? `Testing… ${left} s` : "Test the connection"}
          </button>
          {tested && (
            <span className={tested.ok ? "settings-tested ok" : "settings-tested bad"} role="status">
              {tested.ok
                ? `Connected: the model answered in ${tested.seconds ?? "?"} s.` +
                  (tested.images === null ? "" : tested.images ? " It can see images." : " It cannot see images: vision stays off.")
                : tested.error}
            </span>
          )}
        </div>
        <div className="settings-actions">
          <button type="button" className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" disabled={!canSave}>
            {saving ? "Saving…" : current ? "Current settings" : "Save"}
          </button>
        </div>
      </footer>
    </form>
  );
}

/** Options as cards, one chosen at most: none until the user picks one. */
function Choices<T extends string>({
  name,
  label,
  options,
  value,
  onChange,
  compact = false,
}: {
  name: string;
  label: string;
  options: Choice<T>[];
  value: T | null;
  onChange: (value: T) => void;
  compact?: boolean;
}) {
  return (
    <div className={compact ? "settings-choices compact" : "settings-choices"} role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <label key={option.value} className="settings-choice">
          <input
            type="radio"
            name={name}
            value={option.value}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          <strong>{option.label}</strong>
          {option.note && <span>{option.note}</span>}
        </label>
      ))}
    </div>
  );
}

function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <span className="field-control">{children}</span>
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}
