from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.auth_routes import router as auth_router
from backend.app.api.v2_routes import router


app = FastAPI(
    title="霁肤 · Jiderma Clinical Agent",
    version="2.0.0",
    description="面向医生的多模态皮肤病辅助决策、医学证据检索与病例报告服务",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_origin_regex=r"^https?://(?:127\.0\.0\.1|localhost)(?::\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


