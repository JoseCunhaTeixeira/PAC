import logging

from fastapi import APIRouter, HTTPException

from masw.api.jobs import Job, job_manager
from masw.models.computing import AnyComputingConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["run"])


@router.post("/run", status_code=202)
def start_run(config: AnyComputingConfig) -> Job:
    return job_manager.submit(config)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job
