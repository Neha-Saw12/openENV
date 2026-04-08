"""
FastAPI app for the Shopping Agent — powered by OpenEnv.

Uses openenv.core.env_server.http_server.create_app() to expose the
ShoppingEnvironment over HTTP + WebSocket endpoints that any EnvClient
(including ShoppingEnvClient) can consume.

Endpoints auto-provided by OpenEnv:
  POST /reset       → Reset environment
  POST /step        → Execute action
  GET  /state       → Current episode state
  GET  /health      → Health check
  GET  /schema      → Action/Observation schemas
  WS   /ws          → WebSocket persistent session

Custom endpoints added below:
  GET  /             → Web UI
  GET  /profile      → User personality profile
"""

import os
import sys
from pathlib import Path

import uvicorn

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openenv.core.env_server.http_server import create_app
from openenv_models import ShoppingAction, ShoppingObservation
from server.shopping_environment import ShoppingEnvironment
from memory_engine import load_profile

from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# --- Create the OpenEnv app -------------------------------------------------
# Pass the CLASS (factory), not an instance — create_app creates per-session.
app = create_app(
    ShoppingEnvironment,
    ShoppingAction,
    ShoppingObservation,
    env_name="shopping_agent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Custom endpoints -------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the web UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse(content={"message": "Shopping Agent OpenEnv server is running."})


@app.get("/profile")
async def profile():
    """Return the loaded user personality profile summary."""
    prof = load_profile()
    return {
        "personality_summary": prof.personality_summary[:500],
        "preferences": {
            k: getattr(prof, k)
            for k in [
                "price_sensitivity", "quality_preference", "risk_aversion",
                "research_depth", "brand_trust", "exploration_vs_repeat",
                "review_dependence", "return_preference", "decision_speed",
                "discount_sensitivity",
            ]
        },
        "decision_process": prof.decision_process,
        "semantic_conclusions": [
            {"conclusion": c.get("conclusion", ""), "confidence": c.get("confidence", 0)}
            for c in prof.semantic_conclusions[:6]
        ],
        "shopping_goals": prof.shopping_goals[:400],
    }


# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    """Run the server directly."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
