"""What was done to a unit, as its card's "Settings, and why" says it: each step it went through
with the settings it ran with (the run's preset with the unit's own changes on top: the
assistant's retries, from its QC log; a window's, the line's too), and where each comes from,
its history when the retries changed it. Read, never measured."""

from datetime import UTC, datetime
from typing import Any

from masw.io.quality.files import merged, preset_stage
from masw.io.quality.log import LINE, Attempt, QCLog
from masw.io.quality.runs import DEFAULT, GIVEN, PAC
from masw.io.quality.view import Origin, Setting, number
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.presets import make_preset
from sigpipe.masw.runs import RunManifest

# The stages as their steps say them, and their values' units.
_NAMES = {
    "trigger": "Trigger",
    "muting": "Muting",
    "filtering": "Filter",
    "slicing": "Segments",
    "selection": "Selection",
    "whitening": "Whitening",
    "normalization": "Normalization",
    "stacking": "Stacking",
    "dispersion": "Phase shift",
    "image_stacking": "Images stacked",
}
_UNITS = {
    "t0": "s",
    "tmin": "s",
    "tmax": "s",
    "width": "s",
    "length_s": "s",
    "overlap_s": "s",
    "fmin": "Hz",
    "fmax": "Hz",
    "taper_width_Hz": "Hz",
    "vmin": "m/s",
    "vmax": "m/s",
}
# A step every run does the same way.
_ALWAYS = "every run does it"
# From sigpipe c17b3de (2026-09-28 16:46, +02:00) on, a shot's trigger is corrected with its
# muting only; the runs before corrected it whatever the muting.
_TRIGGER_WITH_MUTING = datetime(2026, 9, 28, 14, 46, tzinfo=UTC)


def _value(key: str, value: object) -> str:
    unit = _UNITS.get(key)
    shown = number(value, 4) if isinstance(value, int | float) else str(value)
    return f"{shown} {unit}" if unit else shown


def _values(values: dict[str, Any]) -> str:
    """A stage's values but its method, in words: "fmin 5 Hz, fmax 220 Hz"."""
    return ", ".join(
        f"{key.replace('_', ' ')} {_value(key, value)}"
        for key, value in values.items()
        if key != "method" and value is not None
    )


def _trigger(attempt: Attempt) -> str:
    """Why an attempt ran, in words."""
    if attempt.triggered_by == "initial":
        return "first run"
    if attempt.triggered_by == "backtrack":
        return "an earlier stage ran again"
    return attempt.triggered_by.replace(":", " ").replace("_", " ")


class _Origins:
    """Where a unit's settings of one log stage come from: its own retries (their history), the
    line's (a window's), else the run's (the preset's default, the request, PAC's form)."""

    def __init__(self, manifest: RunManifest, log: QCLog | None, unit: str, log_stage: str) -> None:
        self.log = log
        self.own = [one for one in log.of(unit, log_stage) if one.parameters] if log else []
        self.line = log.latest(LINE, log_stage) if log is not None else None
        self.preset: dict[str, Any] = manifest.preset.model_dump(mode="json")
        self.defaults: dict[str, Any] = make_preset(manifest.preset.mode).model_dump(mode="json")

    def of(self, stage: str) -> tuple[Origin, str]:
        changed = [one for one in self.own if stage in one.parameters]
        if changed:
            history = "; ".join(
                f"attempt {one.attempt} ({_trigger(one)}): "
                + (_values(one.parameters[stage]) or str(one.parameters[stage].get("method")))
                for one in changed
            )
            return "rule", f"the checks changed it on this unit: {history}"
        if self.line is not None and stage in self.line.parameters:
            return "rule", "the checks changed it on the line"
        if self.preset.get(stage) == self.defaults.get(stage):
            return "default", DEFAULT
        return ("given", GIVEN) if self.log is not None else ("pac", PAC)


def _stages(
    manifest: RunManifest, log: QCLog | None, unit: str, log_stage: str, names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """The settings `names` ran with for `unit` at `log_stage`: the run's preset, the line's
    changes (a window's), the unit's own last."""
    own = log.latest(unit, log_stage) if log is not None else None
    line = log.latest(LINE, log_stage) if log is not None and log_stage == "phase_shift" else None
    changes = merged(line.parameters if line else {}, own.parameters if own else {})
    return {
        name: merged(preset_stage(manifest, name), changes.get(name, {}) or {}) for name in names
    }


def _setting(stage: str, values: dict[str, Any], origins: _Origins) -> Setting:
    origin, why = origins.of(stage)
    method = values.get("method")
    return Setting(
        key=stage,
        label=_NAMES.get(stage, stage.replace("_", " ").capitalize()),
        value=str(method) if method is not None else _values(values) or "-",
        detail=_values(values) if method not in (None, "none") else "",
        why=why,
        origin=origin,
    )


def _fixed(key: str, label: str, value: str) -> Setting:
    return Setting(key=key, label=label, value=value, why=_ALWAYS, origin="default")


def record_settings(manifest: RunManifest, log: QCLog | None, name: str) -> tuple[Setting, ...]:
    """What was done to record `name` before any window used it: its trigger corrected (a shot's,
    when muted), detrended, muted, filtered."""
    stages = _stages(manifest, log, name, "preprocessing", ("trigger", "muting", "filtering"))
    origins = _Origins(manifest, log, name, "preprocessing")
    fields = type(manifest.preset).model_fields
    settings: list[Setting] = []
    muted = stages["muting"].get("method", "none") != "none"
    started = manifest.started_at
    before = (started if started.tzinfo else started.replace(tzinfo=UTC)) < _TRIGGER_WITH_MUTING
    if "trigger" in fields:
        t0 = stages["trigger"].get("t0")
        origin, why = origins.of("trigger")
        shifted = muted or before
        settings.append(
            Setting(
                key="trigger",
                label="Trigger",
                value=(_value("t0", t0) if t0 is not None else "its file's") if shifted else "none",
                detail="" if shifted else "not muted: left as recorded",
                why=why,
                origin=origin,
            )
        )
    settings.append(_fixed("detrend", "Detrend", "mean, then linear"))
    if "muting" in fields:
        settings.append(_setting("muting", stages["muting"], origins))
    settings.append(_setting("filtering", stages["filtering"], origins))
    return tuple(settings)


def window_settings(manifest: RunManifest, log: QCLog | None, unit: str) -> tuple[Setting, ...]:
    """What was done to window `unit` to make its dispersion image, in its steps' order: its
    receivers, then per mode, the shots' images stacked (active), their correlations stacked
    (passive-active), or the noise's segments selected, whitened, normalized, correlated and
    stacked (passive); the phase shift."""
    mode = str(manifest.preset.mode)
    names = {
        "active": ("dispersion", "image_stacking"),
        "passive-active": ("stacking", "dispersion"),
        "passive": (
            "slicing",
            "selection",
            "whitening",
            "normalization",
            "stacking",
            "dispersion",
        ),
    }[mode]
    stages = _stages(manifest, log, unit, "phase_shift", ("masw", *names))
    origins = _Origins(manifest, log, unit, "phase_shift")
    length = int(stages["masw"].get("length", 0))
    spacing = manifest.profile.receiver_spacing_m
    origin, why = origins.of("masw")
    settings = [
        Setting(
            key="window",
            label="Window",
            value=f"{number((length - 1) * spacing, 4)} m",
            detail=f"{length} receivers",
            why=why,
            origin=origin,
        )
    ]
    apodized = _fixed("apodize", "Apodized", "hanning, 10 %")
    if mode == "active":
        settings += [
            _setting("dispersion", stages["dispersion"], origins),
            _setting("image_stacking", stages["image_stacking"], origins),
        ]
    elif mode == "passive-active":
        settings += [
            apodized,
            _fixed("correlate", "Correlated", "each shot with the receiver nearest it"),
            _setting("stacking", stages["stacking"], origins),
            _setting("dispersion", stages["dispersion"], origins),
        ]
    else:
        settings += [_setting(name, stages[name], origins) for name in names[:4]]
        settings += [
            apodized,
            _fixed("correlate", "Correlated", "with the first receiver, causal"),
            _setting("stacking", stages["stacking"], origins),
            _setting("dispersion", stages["dispersion"], origins),
        ]
    return tuple(settings)


def inversion_settings_of(
    parameters: InversionParameters, log: QCLog | None, unit: str
) -> tuple[Setting, ...]:
    """What the window's inversion ran with: the sampler and its layers' priors; the checks'
    retries that changed them, their history."""
    retried = [one for one in log.of(unit, "inversion") if one.parameters] if log else []
    history = "; ".join(f"attempt {one.attempt} ({_trigger(one)})" for one in retried)
    origin: Origin = "rule" if retried else ("given" if log is not None else "pac")
    why = (
        f"the checks changed them on this unit: {history}"
        if retried
        else GIVEN
        if log is not None
        else PAC
    )
    if parameters.layering == "free":
        free = parameters.free
        layers = Setting(
            key="layers",
            label="Layers",
            value=f"chosen by the data, at most {free.max_layers}",
            detail=f"interfaces {number(free.depth_min or 0.0, 4)}-"
            f"{number(free.depth_max or 0.0, 4)} m, Vs {number(free.vs_min or 0.0, 4)}-"
            f"{number(free.vs_max or 0.0, 4)} m/s",
            why=why,
            origin=origin,
        )
    else:
        layers = Setting(
            key="layers",
            label="Layers",
            value=f"{parameters.n_layers} given",
            detail="Vs "
            + ", ".join(
                f"{number(one.vs_min, 4)}-{number(one.vs_max, 4)}" for one in parameters.vs_layers
            )
            + " m/s",
            why=why,
            origin=origin,
        )
    return (
        Setting(
            key="sampler",
            label="MCMC",
            value=f"{parameters.n_chains} chains of {parameters.n_iterations:,} iterations",
            detail=f"{parameters.n_burnin_iterations:,} burn-in",
            why=why,
            origin=origin,
        ),
        layers,
        Setting(
            key="drop",
            label="Vs drop",
            value=f"at most {parameters.max_vs_drop:.0%}",
            detail="from a layer to the next",
            why=why,
            origin=origin,
        ),
    )


def petro_settings(model: str | None, log: QCLog | None) -> tuple[Setting, ...]:
    """What the window's petrophysical inversion ran: the Silex model, on its fundamental mode."""
    if not model:
        return ()
    return (
        Setting(
            key="model",
            label="Silex model",
            value=model,
            detail="fitted to the picked fundamental mode",
            why=GIVEN if log is not None else PAC,
            origin="given" if log is not None else "pac",
        ),
    )
