import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from masw.models.inversion import InversionRunConfig
from masw.models.petro_inversion import PetroInversionRunConfig
from masw.models.processing import ProcessingRequest
from masw.runners.computing import WindowError, run_compute
from masw.runners.inversion import run_inversion
from masw.runners.petro_inversion import run_petro_inversion
from sigpipe.masw.runs import RunManifest, Stopped

logger = logging.getLogger(__name__)

type JobKind = Literal["processing", "inversion", "petro_inversion"]


class JobState(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"  # on request: what had finished kept, nothing half-written


class Job(BaseModel):
    id: str
    kind: JobKind
    mode: str | None = None  # a processing job's mode
    target: str  # what it runs on: a processing job's profile, an inversion's run folder
    state: JobState = JobState.RUNNING
    stopping: bool = False  # a stop was asked, and the job has not ended yet
    completed: int = 0
    total: int = 0
    elapsed: float | None = None  # seconds, set when finished
    error: str | None = None
    errors: list[WindowError] = []  # per-window failures
    run: str | None = None  # a processing job's run, <profile>/<run_id>, once it has ended


class JobManager:
    """PAC's jobs, one at a time in a background thread: each can be stopped (see
    sigpipe.masw.runs.stopping), a job not started yet cancelled outright."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._jobs: dict[str, Job] = {}
        self._started: dict[str, float] = {}
        self._stops: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[list[WindowError]]] = {}

    def submit(self, request: ProcessingRequest) -> Job:
        job = self._new_job("processing", request.profile, request.mode.value)
        stop = self._stops[job.id]

        def work() -> list[WindowError]:
            try:
                job.run, errors = run_compute(request, self._progress(job), stop)
            except Stopped as stopped:
                # The windows that finished kept, as a run of their own.
                if isinstance(stopped.kept, RunManifest):
                    job.run = f"{stopped.kept.profile.name}/{stopped.kept.run_id}"
                raise
            return errors

        return self._launch(job, work)

    def submit_inversion(self, config: InversionRunConfig) -> Job:
        job = self._new_job("inversion", config.folder)
        stop = self._stops[job.id]
        return self._launch(job, lambda: run_inversion(config, self._progress(job), stop))

    def submit_petro_inversion(self, config: PetroInversionRunConfig) -> Job:
        job = self._new_job("petro_inversion", config.folder)
        stop = self._stops[job.id]
        return self._launch(job, lambda: run_petro_inversion(config, self._progress(job), stop))

    def stop(self, job_id: str) -> Job | None:
        """Ask job `job_id` to stop: at once, what had finished kept; one not started yet never
        runs. None for an unknown job; a job already ended is left as it is."""
        job = self._jobs.get(job_id)
        if job is None or job.state != JobState.RUNNING:
            return job
        job.stopping = True
        self._stops[job_id].set()
        if self._futures[job_id].cancel():  # still queued behind another job
            job.state = JobState.STOPPED
            job.stopping = False
            job.elapsed = 0.0
            logger.info("Job %s stopped before it started", job_id)
        return job

    def jobs(self) -> list[Job]:
        """Every job of this process, in the order they were submitted."""
        return list(self._jobs.values())

    def _new_job(self, kind: JobKind, target: str, mode: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, mode=mode, target=target)
        self._jobs[job.id] = job
        self._started[job.id] = time.monotonic()
        self._stops[job.id] = threading.Event()
        return job

    def _progress(self, job: Job) -> Callable[[int, int, WindowError | None], None]:
        def on_progress(completed: int, total: int, err: WindowError | None) -> None:
            job.completed = completed
            job.total = total
            if err is not None:
                job.errors.append(err)

        return on_progress

    def _launch(self, job: Job, work: Callable[[], list[WindowError]]) -> Job:
        future = self._executor.submit(work)
        self._futures[job.id] = future
        future.add_done_callback(lambda f: self._finalize(job.id, f))
        logger.info("Submitted %s job %s", job.kind, job.id)
        return job

    def _finalize(self, job_id: str, future: Future[list[WindowError]]) -> None:
        job = self._jobs[job_id]
        if future.cancelled():
            return  # stopped before it started: stop() said so
        job.elapsed = time.monotonic() - self._started[job_id]
        job.stopping = False
        error = future.exception()
        if isinstance(error, Stopped):
            job.state = JobState.STOPPED
            logger.info("Job %s stopped after %.2f s", job_id, job.elapsed)
        elif error is None:
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
