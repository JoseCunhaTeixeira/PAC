"""The processing settings: sigpipe's presets, one schema for PAC's forms and PACo's agent. The
forms start from a preset fitted to the chosen profile, and send back what they changed."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from masw.io.paths import workspace
from masw.models.processing import ProcessingRequest
from sigpipe.masw.presets import PresetError, make_preset, method_defaults, resolve_preset
from sigpipe.masw.profiles import ProcessingMode, ProfileError, load_profile

logger = logging.getLogger(__name__)

router = APIRouter(tags=["config"])


class PresetDefaults(BaseModel):
    values: dict[str, Any]  # the preset, fitted to the profile
    methods: dict[str, dict[str, dict[str, Any]]]  # by stage, each method's own values


@router.get("/presets/{mode}")
def get_preset(mode: ProcessingMode, profile: str) -> PresetDefaults:
    try:
        loaded = load_profile(profile, workspace())
        values = resolve_preset(make_preset(mode), loaded).model_dump(mode="json")
        return PresetDefaults(values=values, methods=method_defaults(mode, loaded))
    except (ProfileError, PresetError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def check_request(request: ProcessingRequest) -> None:
    """Raises a 422 with every problem, one per line, before any work."""
    try:
        resolve_preset(
            make_preset(request.mode, request.overrides), load_profile(request.profile, workspace())
        )
    except (ProfileError, PresetError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/config")
def validate_config(request: ProcessingRequest) -> dict[str, object]:
    check_request(request)
    logger.info("Validated %s config", request.mode.value)
    return {"valid": True, "mode": request.mode}
