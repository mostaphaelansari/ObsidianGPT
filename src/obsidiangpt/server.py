from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from obsidiangpt.config import RunConfig
from obsidiangpt.pipeline import run_goal
from obsidiangpt.models import RetrievedChunk

logger = logging.getLogger(__name__)

app = FastAPI(title="ObsidianGPT Web API")

# Mount the static directory for the frontend UI
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory mock state for the frontend session
SESSION_HISTORY = []
USER_SETTINGS = {
    "model": "qwen3:1.7b",  # Default local model
    "top_k_chunks": 5
}


class RunRequest(BaseModel):
    goal: str = Field(..., description="The user's goal (e.g., 'What is SQL?')")
    vault_path: str | None = Field(None, description="Optional override for vault path")


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

@app.get("/api/vault")
def api_get_vault():
    load_dotenv()
    vault_path = os.environ.get("VAULT_PATH") or "."
    files = []
    try:
        if Path(vault_path).exists():
            files = [p.name for p in Path(vault_path).rglob("*.md")]
    except Exception:
        pass
    return {"files": sorted(files)}

@app.get("/api/settings")
def api_get_settings():
    return USER_SETTINGS

@app.post("/api/settings")
def api_update_settings(updates: SettingsUpdate):
    if updates.model is not None:
        USER_SETTINGS["model"] = updates.model
    if updates.top_k_chunks is not None:
        USER_SETTINGS["top_k_chunks"] = updates.top_k_chunks
    return USER_SETTINGS


@app.post("/api/run", response_model=RunResponse)
def api_run_goal(req: RunRequest):
    """Execute the ObsidianGPT pipeline for a given goal."""
    try:
        # Load environment via dotenv
        load_dotenv()
        
        # Configure the run
        vault_path = req.vault_path or os.environ.get("VAULT_PATH") or "."
        cfg = RunConfig(vault_root=Path(vault_path), dry_run=True)  # By default, don't write files for the Web UI

        # Run pipeline using active settings
        cfg.top_k_chunks = USER_SETTINGS.get("top_k_chunks", 5)
        # Note: run_goal currently hardcodes or uses default reason() model args.
        # But we'll leave it simple for the UI emulation.
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
