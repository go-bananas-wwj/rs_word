from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rs_words.cli import create as cli_create
from rs_words.config import OUTPUT_DIR, PATCH_BANK_DIR, WEB_DIR

app = FastAPI(title="rs-words 河流汉字")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


def _sanitize_filename(text: str) -> str:
    """Return a filesystem-safe basename from user input."""
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)
    safe = safe.strip("_.-")[:50]
    return safe or "output"


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/create")
def create_api(text: str = Form(...), font_size: int = Form(256)):
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if font_size <= 0:
        raise HTTPException(status_code=400, detail="font_size must be positive")

    safe_text = _sanitize_filename(text)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{safe_text}.png"
    meta_output = OUTPUT_DIR / f"{safe_text}.json"

    try:
        cli_create(
            text=text,
            output=output,
            font_path=None,
            patch_bank_dir=PATCH_BANK_DIR,
            font_size=font_size,
            k=5,
            meta_output=meta_output,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    image_bytes = Path(output).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    meta = json.loads(Path(meta_output).read_text(encoding="utf-8"))

    return JSONResponse({"image": f"data:image/png;base64,{encoded}", "meta": meta})
