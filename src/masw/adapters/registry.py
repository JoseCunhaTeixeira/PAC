from typing import Callable

from sigproc.base import Pipeline

from masw.adapters.active import build_active_pipeline
from masw.adapters.active_passive import build_active_passive_pipeline
from masw.adapters.passive import build_passive_pipeline
from masw.models.modes import ProcessingMode

PIPELINE_BUILDERS: dict[ProcessingMode, Callable[..., Pipeline]] = {
    ProcessingMode.ACTIVE: build_active_pipeline,
    ProcessingMode.PASSIVE: build_passive_pipeline,
    ProcessingMode.ACTIVE_PASSIVE: build_active_passive_pipeline,
}
