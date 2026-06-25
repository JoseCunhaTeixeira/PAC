from collections.abc import Callable

from sigpipe.base import Pipeline

from masw.adapters.active import build_active_pipeline
from masw.adapters.passive import build_passive_pipeline
from masw.adapters.passive_active import build_passive_active_pipeline
from masw.models.modes import ProcessingMode

PIPELINE_BUILDERS: dict[ProcessingMode, Callable[..., Pipeline]] = {
    ProcessingMode.ACTIVE: build_active_pipeline,
    ProcessingMode.PASSIVE: build_passive_pipeline,
    ProcessingMode.PASSIVE_ACTIVE: build_passive_active_pipeline,
}
