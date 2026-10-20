
from contextlib import asynccontextmanager

from ai_analtytics_models.routers.users import router as users_router

from fastapi import FastAPI
from sqlalchemy import text

from ai_analtytics_models.config import settings
from ai_analtytics_models.database import engine

from ai_analtytics_models.routers.datasets import router as datasets_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(
    title="AI Data Analyst Agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(users_router)
app.include_router(datasets_router)


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
