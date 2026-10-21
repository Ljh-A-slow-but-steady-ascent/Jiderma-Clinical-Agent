from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.app.auth import get_auth_repository, login, require_user, token_hash
from backend.app.database import CaseRepository


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(StrictModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)


router = APIRouter(prefix="/v1/auth", tags=["authentication"])


@router.post("/login")
async def auth_login(payload: LoginRequest, repository: CaseRepository = Depends(get_auth_repository)) -> dict:
    result = login(repository, payload.username, payload.password)
    if result is None:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    token, user = result
    return {"access_token": token, "token_type": "bearer", "expires_in": 12 * 3600, "user": user}


@router.get("/me")
async def auth_me(user: dict = Depends(require_user)) -> dict:
    return user


@router.post("/logout")
async def auth_logout(
    authorization: str | None = Header(default=None),
    user: dict = Depends(require_user),
    repository: CaseRepository = Depends(get_auth_repository),
) -> dict:
    del user
    if authorization and authorization.startswith("Bearer "):
        repository.delete_auth_session(token_hash(authorization[7:].strip()))
    return {"status": "logged_out"}
