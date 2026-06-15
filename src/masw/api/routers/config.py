import logging

from fastapi import APIRouter

from masw.models.computing import AnyComputingConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["config"])


@router.post("/config")
def validate_config(config: AnyComputingConfig) -> dict:
    logger.info("Validated %s config", config.mode.value)
    return {"valid": True, "mode": config.mode}
