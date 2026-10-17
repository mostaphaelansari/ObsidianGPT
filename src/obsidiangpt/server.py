from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# Load .env at startup so env vars are populated before any lazy defaults
# in config.py (or anywhere else) are evaluated at import / request time.
load_dotenv()

from obsidiangpt.config import RunConfig
from obsidiangpt.llm import active_provider, resolve_model
from obsidiangpt.pipeline import run_goal
from obsidiangpt.models import RetrievedChunk

logger = logging.getLogger(__name__)

app = FastAPI(title="ObsidianGPT Web API")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

# Mount the static directory for the frontend UI
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory mock state for the frontend session
SESSION_HISTORY = []
USER_SETTINGS = {
    "model": "",  # Empty = the active provider's default model
    "top_k_chunks": 5,
    "vault_path": os.environ.get("VAULT_PATH", ""),
}


class RunRequest(BaseModel):
    goal: str = Field(..., description="The user's goal (e.g., 'What is SQL?')")
    vault_path: str | None = Field(None, description="Optional override for vault path")


class VaultConnectRequest(BaseModel):
    path: str = Field(..., description="Absolute path to a local Obsidian vault folder")


def _browse_for_folder() -> str | None:
    """Pop a native OS folder-picker on the machine running this server.

    Only meaningful when the server is running on the same machine as the
    person using the UI (the standard setup for this app — a local tool,
    not a hosted multi-user service).
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askdirectory(title="Select your Obsidian vault folder")
    finally:
        root.destroy()
    return path or None


class RunResponse(BaseModel):
    subtasks: list[str]
    chunks: list[dict]
    markdown_body: str
    output_path: str | None


class SettingsUpdate(BaseModel):
    model: str | None = None
    top_k_chunks: int | None = None
    
@app.get("/")
def read_root():
    """Serve the main index.html file."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>ObsidianGPT UI</h1><p>index.html not found in static folder.</p>")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/api/history")
def api_get_history():
    return {"history": SESSION_HISTORY}

def _active_vault_path() -> str:
    return USER_SETTINGS.get("vault_path") or os.environ.get("VAULT_PATH") or "."


@app.get("/api/vault")
def api_get_vault():
    vault_path = _active_vault_path()
    files = []
    try:
        if Path(vault_path).exists():
            files = [p.name for p in Path(vault_path).rglob("*.md")]
    except Exception:
        pass
    return {"path": vault_path, "files": sorted(files)}


@app.post("/api/vault/browse")
def api_browse_vault():
    """Open a native folder-picker on the server's machine and return the choice."""
    try:
        path = _browse_for_folder()
    except Exception as e:
        logger.exception("Folder picker failed")
        raise HTTPException(
            status_code=501,
            detail=f"Native folder picker unavailable on this server ({e}). Paste the path manually instead.",
        )
    return {"path": path}


@app.post("/api/vault/connect")
def api_connect_vault(req: VaultConnectRequest):
    """Point the agent at a local Obsidian vault folder on this machine."""
    vault_path = Path(req.path).expanduser()
    if not vault_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {vault_path}")

    vault_path = vault_path.resolve()
    md_files = sorted(p.name for p in vault_path.rglob("*.md"))
    is_obsidian_vault = (vault_path / ".obsidian").exists()

    USER_SETTINGS["vault_path"] = str(vault_path)

    return {
        "path": str(vault_path),
        "files": md_files,
        "file_count": len(md_files),
        "is_obsidian_vault": is_obsidian_vault,
    }

def _settings_view() -> dict:
    provider = active_provider()
    return {**USER_SETTINGS, "provider": provider, "default_model": resolve_model(provider)}


@app.get("/api/settings")
def api_get_settings():
    return _settings_view()

@app.post("/api/settings")
def api_update_settings(updates: SettingsUpdate):
    if updates.model is not None:
        USER_SETTINGS["model"] = updates.model
    if updates.top_k_chunks is not None:
        USER_SETTINGS["top_k_chunks"] = updates.top_k_chunks
    return _settings_view()


@app.post("/api/run", response_model=RunResponse)
def api_run_goal(req: RunRequest):
    """Execute the ObsidianGPT pipeline for a given goal."""
    try:
        # Configure the run
        vault_path = req.vault_path or _active_vault_path()
        cfg = RunConfig(vault_root=Path(vault_path), dry_run=True)  # By default, don't write files for the Web UI

        # Run pipeline using active settings
        cfg.top_k_chunks = USER_SETTINGS.get("top_k_chunks", 5)
        cfg.model = USER_SETTINGS.get("model") or None
        plan, chunks, synthesis, write_result = run_goal(cfg, req.goal)

        # Format retrieved chunks for the frontend
        chunks_data = [
            {"path": str(c.path.name), "excerpt": c.excerpt, "score": c.score}
            for c in chunks
        ]
        
        resp = RunResponse(
            subtasks=[s.question for s in plan.subtasks],
            chunks=chunks_data,
            markdown_body=synthesis.markdown_body,
            output_path=str(write_result.path) if write_result and write_result.path else None
        )
        
        # Save to memory history
        SESSION_HISTORY.insert(0, {
            "goal": req.goal,
            "markdown": synthesis.markdown_body[:200] + "..." if len(synthesis.markdown_body) > 200 else synthesis.markdown_body
        })

        return resp

    except Exception as e:
        logger.exception("Error executing pipeline")
        raise HTTPException(status_code=500, detail=str(e))
