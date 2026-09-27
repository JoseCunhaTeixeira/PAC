import logging

from fastapi import APIRouter, HTTPException

from masw.api.jobs import Job, job_manager
from masw.api.routers.config import check_request
from masw.models.processing import ProcessingRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["run"])


@router.post("/run", status_code=202)
def start_run(request: ProcessingRequest) -> Job:
    check_request(request)
    return job_manager.submit(request)


@router.get("/jobs")
def list_jobs() -> list[Job]:
    """Every job since the server started: a page finds the one it left running."""
    return job_manager.jobs()


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str) -> Job:
    """Stop the job at once: what had finished kept, nothing half-written left; a job already
    ended is returned as it is."""
    job = job_manager.stop(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job
