"""FastAPI service.   uvicorn api:app --host 0.0.0.0 --port 8000

Every endpoint except /health needs `Authorization: Bearer <token>`. Groups are looked up on the
server from the database for each request; the client cannot supply or widen them.
Not executed in the sandbox this was written in (no fastapi installed, no network): run
`python -m unittest tests.test_api` after `pip install -r requirements.txt -r requirements-dev.txt`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from enterprise_rag.access import Principal
from enterprise_rag.db import Database
from enterprise_rag.factory import build_stack
from enterprise_rag.pipeline import MAX_FILE_BYTES
from enterprise_rag.service import AccessDenied, KnowledgeService, NotFound
from enterprise_rag.settings import Settings

bearer = HTTPBearer(auto_error=False)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class PermissionsRequest(BaseModel):
    access_groups: list[str] = Field(min_length=1)


def create_app(service: KnowledgeService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            settings = Settings.from_env()
            db = Database(settings.database_url)
            db.init_schema()
            built = KnowledgeService(build_stack(settings), db)
            built.sync_document_records()
            app.state.service = built
        else:
            app.state.service = service
        yield

    app = FastAPI(title="Enterprise Knowledge Intelligence System", version="1.0.0", lifespan=lifespan)

    def principal(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
        found = app.state.service.db.authenticate(credentials.credentials) if credentials else None
        if found is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing token.",
                                headers={"WWW-Authenticate": "Bearer"})
        return found

    def guard(call):
        try:
            return call()
        except NotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None
        except AccessDenied as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None

    @app.get("/health")
    def health() -> dict[str, Any]:
        svc = app.state.service
        return {"status": "ok", "chunks": len(svc.index.chunks), "embedder": svc.stack.embedder.fingerprint}

    @app.get("/me")
    def me(user: Principal = Depends(principal)) -> dict[str, Any]:
        return {"user_id": user.user_id, "groups": sorted(user.groups), "is_admin": user.is_admin}

    @app.post("/ask")
    def ask(body: AskRequest, user: Principal = Depends(principal)) -> dict[str, Any]:
        return guard(lambda: app.state.service.ask(user, body.question, body.session_id))

    @app.post("/sessions")
    def create_session(user: Principal = Depends(principal)) -> dict[str, str]:
        return {"session_id": app.state.service.create_session(user)}

    @app.get("/sessions/{session_id}/history")
    def history(session_id: str, user: Principal = Depends(principal)) -> list[dict[str, Any]]:
        return guard(lambda: app.state.service.history(user, session_id))

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str, user: Principal = Depends(principal)) -> None:
        guard(lambda: app.state.service.delete_session(user, session_id))

    @app.get("/documents")
    def documents(user: Principal = Depends(principal)) -> list[dict[str, Any]]:
        docs = app.state.service.list_documents(user)
        return docs if user.is_admin else [{k: v for k, v in d.items() if k != "access_groups"} for d in docs]

    @app.post("/documents", status_code=201)
    async def upload(file: UploadFile = File(...), metadata: str = Form(...),
                     user: Principal = Depends(principal)) -> dict[str, Any]:
        if not user.is_admin:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator rights are required.")
        data = await file.read(MAX_FILE_BYTES + 1)
        try:
            record = json.loads(metadata)
        except ValueError:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "metadata must be a JSON object.") from None
        return guard(lambda: app.state.service.ingest_upload(user, file.filename or "upload", data, record))

    @app.patch("/documents/{document_id}/permissions")
    def permissions(document_id: str, body: PermissionsRequest, user: Principal = Depends(principal)) -> dict[str, Any]:
        return guard(lambda: app.state.service.update_permissions(user, document_id, body.access_groups))

    @app.delete("/documents/{document_id}", status_code=204)
    def delete_document(document_id: str, user: Principal = Depends(principal)) -> None:
        guard(lambda: app.state.service.delete_document(user, document_id))

    return app


app = create_app()
