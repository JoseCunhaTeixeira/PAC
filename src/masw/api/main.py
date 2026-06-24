import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import version as _pkg_version

import matplotlib

# Force the non-interactive Agg backend before any module imports pyplot.
# FastAPI runs synchronous endpoints (e.g. dispersion picking, which plots
# and saves a PNG on every pick) in a worker thread, not the process's main
# thread. matplotlib's default backend on this machine is TkAgg, which
# creates Tk objects tied to whichever thread made them; when those objects
# are later garbage-collected from a different thread (or after no Tk main
# loop is running), Tk raises "main thread is not in main loop". Agg has no
# GUI/main-loop concept, so it has no such thread affinity.
matplotlib.use("Agg")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from masw.api.routers import (
    acquisitions,
    config,
    dispersion_images,
    gather,
    inversion,
    run,
    windows,
)
from masw.logging_config import setup_logging

logger = logging.getLogger(__name__)

__version__ = _pkg_version("PAC")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    setup_logging()
    logger.info("MASW API starting")
    yield
    logger.info("MASW API shutting down")


app = FastAPI(title="MASW API", version=__version__, lifespan=lifespan)

_cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(acquisitions.router)
app.include_router(config.router)
app.include_router(run.router)
app.include_router(windows.router)
app.include_router(gather.router)
app.include_router(dispersion_images.router)
app.include_router(inversion.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def get_version() -> dict[str, str]:
    return {"version": __version__}
