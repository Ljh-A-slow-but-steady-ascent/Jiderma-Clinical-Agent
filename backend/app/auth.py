from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta

from fastapi import Depends, Header, HTTPException

from backend.app.database import CaseRepository


PBKDF2_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def bootstrap_user(repository: CaseRepository) -> None:
    username = os.getenv("JIDERMA_ADMIN_USERNAME", "").strip()
    password = os.getenv("JIDERMA_ADMIN_PASSWORD", "").strip()
    if username and password and repository.get_user_by_username(username) is None:
        repository.create_user(username, hash_password(password), os.getenv("JIDERMA_ADMIN_NAME", "霁肤医生"))


def login(repository: CaseRepository, username: str, password: str) -> tuple[str, dict] | None:
    bootstrap_user(repository)
    user = repository.get_user_by_username(username)
    if user is None or not user.active or not verify_password(password, user.password_hash):
        return None
    token = secrets.token_urlsafe(40)
    repository.create_auth_session(token_hash(token), user.id, datetime.utcnow() + timedelta(hours=12))
    return token, repository.user_public(user)


_auth_repository = CaseRepository()
bootstrap_user(_auth_repository)


async def get_auth_repository() -> CaseRepository:
    return _auth_repository


async def require_user(authorization: str | None = Header(default=None)) -> dict:
    if os.getenv("DERMAGUARD_AUTH_DISABLED", "0") == "1":
        return {"id": 0, "username": "test", "full_name": "测试医生", "role": "clinician"}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    user = _auth_repository.get_session_user(token_hash(authorization[7:].strip()), datetime.utcnow())
    if user is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    return _auth_repository.user_public(user)


CurrentUser = dict


async def current_user(user: dict = Depends(require_user)) -> dict:
    return user
