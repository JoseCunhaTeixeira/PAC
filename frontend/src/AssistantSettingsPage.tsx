import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { API } from "./api";
import { Callout, Card, Fields, Page, Segmented, SelectField, type SegmentOption } from "./components/kit";
import { ArrowRightIcon, CpuIcon, EyeIcon, LockIcon, SlidersIcon, SparklesIcon } from "./components/icons";
import "./AssistantSettingsPage.css";

// The assistant's settings (GET/PUT /agent/settings): which of PACo's agents the chat runs, the
// model both talk to, and whether figures may reach it. The API key is never sent back: the page
// knows only whether one is set.

type Mode = "guided" | "autonomous";

// A model provider PACo talks to, and what it needs (GET /agent/providers).
interface Provider {
  name: string;
  label: string;
  address: "needed" | "optional";
  key: "needed" | "optional" | "none";
  region: boolean;
  project: boolean;
  context: number; // the context assumed when none is set, in tokens
}

interface Settings {
  mode: Mode;
  provider: string;
  model: string;
  base_url: string;
  region: string;
  project: string;
  context: number | null;
  vision: boolean;
  key_set: boolean;
}

// A connection test (POST /agent/settings/test): one short call, never a figure.
interface Tested {
  ok: boolean;
  reply: string | null;
  seconds: number | null;
  images: boolean | null;
  error: string | null;
}

// The context the autonomous geophysicist needs, in tokens.
const LONG_CONTEXT = 131_072;

const MODES: SegmentOption<Mode>[] = [
  { value: "guided", label: "Guided workflow", title: "PACo leads the model step by step" },
  { value: "autonomous", label: "Autonomous geophysicist", title: "The model chooses and tries its own settings" },
];

const VISION: SegmentOption<"off" | "on">[] = [
  { value: "off", label: "Off", title: "Figures stay on this machine" },
  { value: "on", label: "On", title: "Figures go to the model's provider" },
];

// What each agent does, and what it asks of the model.
const MODE_SAID: Record<Mode, ReactNode> = {
  guided: (
    <>
      <p>
        PACo leads the model through its workflow, one tool at a time: the model reads your request and calls PACo&apos;s
        tools in order, and the quality gates check each stage and fix what they can.
      </p>
      <p>
        Made for a small model on your own GPU (Qwen3-14B served by vLLM, a context of 16k tokens): your data never leave
        your machines, and an answer costs nothing beyond the GPU.
      </p>
    </>
  ),
  autonomous: (
    <>
      <p>
        The model works as a geophysicist: it surveys the line, chooses the windows, the shots and the images, runs its
        own experiments and judges them with PACo&apos;s quality gates, decides every window, then inverts. Its journal of
        hypotheses, experiments and decisions is on the run&apos;s card in Visualization.
      </p>
      <p>
        It needs a frontier model with a <b>long context, 128k tokens or more</b>, and it uses many more tokens than the
        guided workflow: hundreds of thousands for a line, which a paid API bills.
      </p>
    </>
  ),
};

async function detail(res: Response): Promise<string> {
  const body = await res.json().catch(() => null);
  return typeof body?.detail === "string" ? body.detail : `HTTP ${res.status}`;
}

function TextField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  type = "text",
  icon,
}: {
  label: ReactNode;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  hint?: ReactNode;
  type?: "text" | "password" | "url";
  icon?: ReactNode;
}) {
  return (
    <label className="field">
      <span className="field-label">
        {icon}
        {label}
      </span>
      <span className="field-control">
        <input
          type={type}
          value={value}
          placeholder={placeholder}
          autoComplete={type === "password" ? "new-password" : "off"}
          spellCheck={false}
          onChange={(e) => onChange(e.target.value)}
        />
      </span>
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export default function AssistantSettingsPage() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [saved, setSaved] = useState<Settings | null>(null);
  const [form, setForm] = useState<Settings | null>(null);
  // A new key typed (empty: keep the one set), or the one set removed.
  const [key, setKey] = useState("");
  const [removeKey, setRemoveKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [said, setSaid] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  const [testing, setTesting] = useState(false);
  const [tested, setTested] = useState<Tested | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetch(`${API}/agent/settings`), fetch(`${API}/agent/providers`)])
      .then(async ([settings, offered]) => {
        if (!settings.ok) throw new Error(await detail(settings));
        if (!offered.ok) throw new Error(await detail(offered));
        const found = (await settings.json()) as Settings;
        setSaved(found);
        setForm(found);
        setProviders((await offered.json()) as Provider[]);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const provider = useMemo(
    () => providers.find((one) => one.name === form?.provider) ?? null,
    [providers, form?.provider],
  );
  const changed =
    form !== null && saved !== null && (JSON.stringify(form) !== JSON.stringify(saved) || key !== "" || removeKey);
  // Another provider than the one saved: the key set is that one's, not sent to this one.
  const otherProvider = form !== null && saved !== null && form.provider !== saved.provider;

  function set<K extends keyof Settings>(name: K, value: Settings[K]) {
    setForm((previous) => (previous ? { ...previous, [name]: value } : previous));
    setSaid(null);
    setTested(null);
  }

  // Another provider: the settings it needs, to fill (back to the one saved: its own again).
  function chooseProvider(name: string) {
    setForm((previous) => {
      if (!previous || !saved) return previous;
      const own =
        name === saved.provider
          ? { model: saved.model, base_url: saved.base_url, region: saved.region, project: saved.project, context: saved.context }
          : { model: "", base_url: "", region: "", project: "", context: null };
      return { ...previous, provider: name, ...own };
    });
    setKey("");
    setRemoveKey(false);
    setSaid(null);
    setTested(null);
  }

  // The settings as the backend takes them: the key typed, none to remove it, or the one set
  // (never another provider's).
  function body(settings: Settings) {
    return {
      mode: settings.mode,
      provider: settings.provider,
      model: settings.model,
      base_url: settings.base_url,
      region: settings.region,
      project: settings.project,
      context: settings.context,
      vision: settings.vision,
      api_key: removeKey || (saved !== null && settings.provider !== saved.provider) ? key : key || null,
    };
  }

  async function save() {
    if (!form) return;
    setSaving(true);
    setSaid(null);
    try {
      const res = await fetch(`${API}/agent/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body(form)),
      });
      if (!res.ok) throw new Error(await detail(res));
      const found = (await res.json()) as Settings;
      setSaved(found);
      setForm(found);
      setKey("");
      setRemoveKey(false);
      setSaid({ tone: "success", text: "Saved. The next conversation runs with these settings; those open keep theirs." });
    } catch (err) {
      setSaid({ tone: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setSaving(false);
    }
  }

  async function test() {
    if (!form) return;
    setTesting(true);
    setTested(null);
    try {
      const res = await fetch(`${API}/agent/settings/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body(form)),
      });
      if (!res.ok) throw new Error(await detail(res));
      setTested((await res.json()) as Tested);
    } catch (err) {
      setTested({ ok: false, reply: null, seconds: null, images: null, error: err instanceof Error ? err.message : String(err) });
    } finally {
      setTesting(false);
    }
  }

  const context = form?.context ?? provider?.context ?? null;
  const short = form?.mode === "autonomous" && context !== null && context < LONG_CONTEXT;

  return (
    <Page
      icon={<SlidersIcon size={24} />}
      title="Assistant settings"
      subtitle="Which agent answers on the AI assistant page, and the model it talks to"
      art="assistant"
      actions={
        <Link to="/assistant" className="settings-back">
          Back to the AI assistant <ArrowRightIcon size={13} />
        </Link>
      }
    >
      {error && (
        <Callout tone="error" title="The settings could not be read">
          {error}
        </Callout>
      )}
      {form && (
        <div className="settings-page">
          <Card icon={<SparklesIcon size={17} />} title="Agent" hint="Both agents run PACo's tools on PAC's folders">
            <Segmented label="Agent" value={form.mode} options={MODES} onChange={(mode) => set("mode", mode)} />
            <div className="settings-said">{MODE_SAID[form.mode]}</div>
            {short && (
              <Callout tone="warn" title="This model's context is too short for the geophysicist">
                {context?.toLocaleString("en-GB")} tokens{form.context === null ? ", as assumed for this provider" : ""}:
                choose a frontier model, or set the context your server serves (vLLM&apos;s --max-model-len) to 128k
                tokens or more.
              </Callout>
            )}
          </Card>

          <Card icon={<CpuIcon size={17} />} title="Model" hint="Any provider you hold a key or a cloud account for">
            <Fields min={260}>
              <SelectField label="Provider" value={form.provider} onChange={chooseProvider}>
                {providers.length === 0 && <option value={form.provider}>{form.provider}</option>}
                {providers.map((one) => (
                  <option key={one.name} value={one.name}>
                    {one.label}
                  </option>
                ))}
              </SelectField>
              <TextField
                label="Model"
                value={form.model}
                onChange={(value) => set("model", value)}
                placeholder={
                  form.provider === "vllm"
                    ? "Qwen/Qwen3-14B-FP8"
                    : form.provider.startsWith("anthropic")
                      ? "claude-opus-5-5"
                      : "the model's name at the provider"
                }
                hint={form.provider === "azure-openai" ? "Your deployment's name" : undefined}
              />
              {provider?.key !== "none" && (
                <TextField
                  label="API key"
                  type="password"
                  icon={<LockIcon size={13} />}
                  value={key}
                  onChange={(value) => {
                    setKey(value);
                    setRemoveKey(false);
                    setSaid(null);
                    setTested(null);
                  }}
                  placeholder={
                    removeKey
                      ? "removed when saved"
                      : form.key_set && !otherProvider
                        ? "set: leave empty to keep it"
                        : provider?.key === "optional"
                          ? "none"
                          : "your key"
                  }
                  hint={
                    form.provider === "anthropic-bedrock"
                      ? "A Bedrock API key, or none to use this machine's AWS credentials"
                      : "Kept on this machine, readable by PAC alone; never shown again"
                  }
                />
              )}
              {provider && (
                <TextField
                  label={provider.address === "needed" ? "Server address" : "Address (optional)"}
                  type="url"
                  value={form.base_url}
                  onChange={(value) => set("base_url", value)}
                  placeholder={
                    provider.address === "needed"
                      ? form.provider === "azure-openai" || form.provider === "anthropic-foundry"
                        ? "your resource's endpoint"
                        : "http://gpu-host:8001/v1"
                      : "the provider's own"
                  }
                />
              )}
              {provider?.region && (
                <TextField
                  label="Region"
                  value={form.region}
                  onChange={(value) => set("region", value)}
                  placeholder={form.provider === "anthropic-bedrock" ? "us-east-1" : "global"}
                />
              )}
              {provider?.project && (
                <TextField
                  label="Google Cloud project"
                  value={form.project}
                  onChange={(value) => set("project", value)}
                  placeholder="your project's id"
                />
              )}
              <label className="field">
                <span className="field-label">Context</span>
                <span className="field-control">
                  <input
                    type="number"
                    min={1}
                    step={1024}
                    value={form.context ?? ""}
                    placeholder={provider ? `auto: ${provider.context.toLocaleString("en-GB")}` : "auto"}
                    onChange={(e) => set("context", e.target.value === "" ? null : Math.max(1, Math.round(Number(e.target.value))))}
                  />
                  <span className="field-unit">tokens</span>
                </span>
                <span className="field-hint">As the model&apos;s server serves it; leave empty for the provider&apos;s usual</span>
              </label>
            </Fields>
            {provider?.key === "none" && (
              <p className="settings-note">
                Vertex AI takes this machine&apos;s Google credentials: run <code>gcloud auth application-default login</code>, or
                set <code>GOOGLE_APPLICATION_CREDENTIALS</code>, where PAC&apos;s backend runs.
              </p>
            )}
            <div className="settings-row">
              <button type="button" className="secondary" onClick={() => void test()} disabled={testing || !form.model}>
                {testing ? "Testing…" : "Test the connection"}
              </button>
              {form.key_set && !otherProvider && provider?.key !== "none" && (
                <button
                  type="button"
                  className="ghost"
                  onClick={() => {
                    setRemoveKey(true);
                    setKey("");
                    setSaid(null);
                  }}
                  disabled={removeKey}
                >
                  Remove the key
                </button>
              )}
              {tested && (
                <span className={tested.ok ? "settings-tested ok" : "settings-tested bad"} role="status">
                  {tested.ok
                    ? `The model answered in ${tested.seconds ?? "?"} s${tested.reply ? `: “${tested.reply}”` : ""}.` +
                      (tested.images === null
                        ? ""
                        : tested.images
                          ? " It takes images."
                          : " It takes no images: vision would be switched off.")
                    : tested.error}
                </span>
              )}
            </div>
          </Card>

          <Card icon={<EyeIcon size={17} />} title="Vision" hint="Whether the agent may look at the figures">
            <Segmented
              label="Vision"
              value={form.vision ? "on" : "off"}
              options={VISION}
              onChange={(value) => set("vision", value === "on")}
            />
            <Callout tone={form.vision ? "warn" : "info"} title={form.vision ? "Figures go to the model's provider" : "Figures stay on this machine"}>
              With vision on, the figures the geophysicist looks at (records, dispersion images, curves) are sent to the
              model&apos;s provider with its requests. Leave it off when your data must stay on your machines, or the provider
              is not approved for them: the agent then reads tables and measures instead. The guided workflow never sends
              figures.
            </Callout>
          </Card>

          <div className="settings-row settings-save">
            <button type="button" onClick={() => void save()} disabled={saving || !changed}>
              {saving ? "Saving…" : "Save"}
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => {
                setForm(saved);
                setKey("");
                setRemoveKey(false);
                setSaid(null);
                setTested(null);
              }}
              disabled={saving || !changed}
            >
              Discard the changes
            </button>
          </div>
          {said && <Callout tone={said.tone}>{said.text}</Callout>}
        </div>
      )}
    </Page>
  );
}
