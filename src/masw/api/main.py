import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from masw.api.routers import acquisitions, config, run, windows
from masw.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("MASWvicorn masw.api.main:app --reload API starting")
    yield
    logger.info("MASW API shutting down")


app = FastAPI(title="MASW API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(acquisitions.router)
app.include_router(config.router)
app.include_router(run.router)
app.include_router(windows.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
