import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum

from pydantic import BaseModel

from masw.models.inversion import InversionRunConfig
from masw.models.petro_inversion import PetroInversionRunConfig
from masw.models.processing import ProcessingRequest
from masw.runners.computing import WindowError, run_compute
from masw.runners.inversion import run_inversion
from masw.runners.petro_inversion import run_petro_inversion

logger = logging.getLogger(__name__)


class JobState(StrEnum):
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
    errors: list[WindowError] = []  # per-window failures
    run: str | None = None  # a processing job's run, <profile>/<run_id>, once it has ended


class JobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._jobs: dict[str, Job] = {}
        self._started: dict[str, float] = {}

    def submit(self, request: ProcessingRequest) -> Job:
        job = self._new_job()

        def work() -> list[WindowError]:
            job.run, errors = run_compute(request, self._progress(job))
            return errors

        return self._launch(job, work, "processing")

    def submit_inversion(self, config: InversionRunConfig) -> Job:
        job = self._new_job()
        return self._launch(job, lambda: run_inversion(config, self._progress(job)), "inversion")

    def submit_petro_inversion(self, config: PetroInversionRunConfig) -> Job:
        job = self._new_job()
        return self._launch(
            job, lambda: run_petro_inversion(config, self._progress(job)), "petro inversion"
        )

    def _new_job(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job
        self._started[job.id] = time.monotonic()
        return job

    def _progress(self, job: Job) -> Callable[[int, int, WindowError | None], None]:
        def on_progress(completed: int, total: int, err: WindowError | None) -> None:
            job.completed = completed
            job.total = total
            if err is not None:
                job.errors.append(err)

        return on_progress

    def _launch(self, job: Job, work: Callable[[], list[WindowError]], kind: str) -> Job:
        future = self._executor.submit(work)
        future.add_done_callback(lambda f: self._finalize(job.id, f))
        logger.info("Submitted %s job %s", kind, job.id)
        return job

    def _finalize(self, job_id: str, future: Future[list[WindowError]]) -> None:
        job = self._jobs[job_id]
        job.elapsed = time.monotonic() - self._started[job_id]
        error = future.exception()
        if error is None:
            # A processing job knows its failed windows once its run has ended; the other jobs
            # report theirs as they go.
            reported = {(one.xmid, one.error_type) for one in job.errors}
            job.errors.extend(
                one for one in future.result() if (one.xmid, one.error_type) not in reported
            )
            job.state = JobState.SUCCEEDED
            logger.info("Job %s succeeded in %.2f s", job_id, job.elapsed)
        else:
            job.state = JobState.FAILED
            job.error = str(error)
            logger.error("Job %s failed after %.2f s: %s", job_id, job.elapsed, error)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


job_manager = JobManager()
