import logging
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum

from pydantic import BaseModel

from masw.models.computing import AnyComputingConfig
from masw.runners.computing import run_compute

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Job(BaseModel):
    id: str
    state: JobState = JobState.RUNNING
    completed: int = 0
    total: int = 0
    elapsed: float | None = None  # seconds, set when finished
    error: str | None = None


class JobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._jobs: dict[str, Job] = {}
        self._started: dict[str, float] = {}

    def submit(self, config: AnyComputingConfig) -> Job:
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job
        self._started[job.id] = time.monotonic()

        def on_progress(completed: int, total: int) -> None:
            job.completed = completed
            job.total = total

        future = self._executor.submit(run_compute, config, on_progress)
        future.add_done_callback(lambda f: self._finalize(job.id, f))

        logger.info("Submitted job %s", job.id)
        return job

    def _finalize(self, job_id: str, future: Future) -> None:
        job = self._jobs[job_id]
        job.elapsed = time.monotonic() - self._started[job_id]
        error = future.exception()
        if error is None:
            job.state = JobState.SUCCEEDED
            logger.info("Job %s succeeded in %.2f s", job_id, job.elapsed)
        else:
            job.state = JobState.FAILED
            job.error = str(error)
            logger.error("Job %s failed after %.2f s: %s", job_id, job.elapsed, error)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


job_manager = JobManager()
