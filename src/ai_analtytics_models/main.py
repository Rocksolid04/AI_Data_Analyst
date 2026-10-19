
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from ai_analtytics_models.config import settings
from ai_analtytics_models.database import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(
    title="AI Data Analyst Agent",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "AI Data Analyst Agent API",
        "status": "running",
    }


@app.get("/health")
async def health():
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))

    return {
        "status": "healthy",
        "database": "connected",
    }
