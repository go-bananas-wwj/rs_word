from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rs_words.cli import create as cli_create
from rs_words.config import OUTPUT_DIR, PATCH_BANK_DIR, WEB_DIR

app = FastAPI(title="rs-words 河流汉字")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

DEFAULT_REVIEW_DATA_DIR = Path(
    "/data2/rs_word_vectors/sentinel10_stroke_imagery_v3_scale1p2"
)
REVIEW_DATA_DIR = Path(os.environ.get("RS_WORDS_REVIEW_DATA_DIR", DEFAULT_REVIEW_DATA_DIR))
REVIEW_SELECTIONS_PATH = Path(
    os.environ.get(
        "RS_WORDS_REVIEW_SELECTIONS",
        str(REVIEW_DATA_DIR / "review_selections.json"),
    )
)

if REVIEW_DATA_DIR.exists():
    app.mount(
        "/review-files",
        StaticFiles(directory=REVIEW_DATA_DIR),
        name="review-files",
    )

Decision = Literal["selected", "rejected", "unreviewed"]


class ReviewSelection(BaseModel):
    chip_id: str
    decision: Decision
    note: str = ""


def _sanitize_filename(text: str) -> str:
    """Return a filesystem-safe basename from user input."""
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)
    safe = safe.strip("_.-")[:50]
    return safe or "output"


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/review", response_class=HTMLResponse)
def review_page():
    return (WEB_DIR / "review.html").read_text(encoding="utf-8")


def _review_manifest_path() -> Path:
    return REVIEW_DATA_DIR / "manifest.jsonl"


def _load_review_rows() -> list[dict]:
    manifest_path = _review_manifest_path()
    if not manifest_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"manifest not found: {manifest_path}",
        )

    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_review_selections() -> dict[str, dict]:
    if not REVIEW_SELECTIONS_PATH.exists():
        return {}
    return json.loads(REVIEW_SELECTIONS_PATH.read_text(encoding="utf-8"))


def _save_review_selections(selections: dict[str, dict]) -> None:
    REVIEW_SELECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = REVIEW_SELECTIONS_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(selections, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(REVIEW_SELECTIONS_PATH)


def _review_image_url(path_text: str) -> str:
    path = Path(path_text)
    try:
        rel_path = path.relative_to(REVIEW_DATA_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"path outside review dir: {path}") from exc
    return f"/review-files/{rel_path.as_posix()}"


@app.get("/api/review/items")
def review_items(
    stroke: str = "all",
    status: str = "all",
    offset: int = 0,
    limit: int = 60,
):
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be non-negative")
    if limit <= 0 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")

    rows = _load_review_rows()
    selections = _load_review_selections()
    strokes = sorted({row["stroke_type"] for row in rows})

    items = []
    status_counts = {"all": 0, "selected": 0, "rejected": 0, "unreviewed": 0}
    stroke_counts = {name: 0 for name in strokes}
    for row in rows:
        chip_id = row["chip_id"]
        selection = selections.get(chip_id, {})
        decision = selection.get("decision", "unreviewed")
        if decision not in ("selected", "rejected"):
            decision = "unreviewed"

        status_counts["all"] += 1
        status_counts[decision] += 1
        stroke_counts[row["stroke_type"]] += 1

        if stroke != "all" and row["stroke_type"] != stroke:
            continue
        if status != "all" and decision != status:
            continue

        candidate = row.get("candidate", {})
        items.append(
            {
                "chip_id": chip_id,
                "stroke_type": row["stroke_type"],
                "rank": row.get("rank"),
                "river": row.get("target_river"),
                "score": candidate.get("score"),
                "cloud_cover": row.get("cloud_cover"),
                "datetime": row.get("datetime"),
                "decision": decision,
                "note": selection.get("note", ""),
                "image_url": _review_image_url(row["preview_path"]),
                "label_url": _review_image_url(row["overlay_path"]),
            }
        )

    page = items[offset : offset + limit]
    return JSONResponse(
        {
            "items": page,
            "total": len(items),
            "offset": offset,
            "limit": limit,
            "strokes": strokes,
            "stroke_counts": stroke_counts,
            "status_counts": status_counts,
            "selection_path": str(REVIEW_SELECTIONS_PATH),
        }
    )


@app.post("/api/review/selection")
def save_review_selection(selection: ReviewSelection):
    rows = _load_review_rows()
    known_ids = {row["chip_id"] for row in rows}
    if selection.chip_id not in known_ids:
        raise HTTPException(status_code=404, detail="unknown chip_id")

    selections = _load_review_selections()
    note = selection.note.strip()
    if selection.decision == "unreviewed" and not note:
        selections.pop(selection.chip_id, None)
    else:
        selections[selection.chip_id] = {
            "decision": selection.decision,
            "note": note,
        }
    _save_review_selections(selections)
    return JSONResponse({"ok": True, "selection_path": str(REVIEW_SELECTIONS_PATH)})


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

    image_bytes = output.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    meta = json.loads(meta_output.read_text(encoding="utf-8"))

    return JSONResponse({"image": f"data:image/png;base64,{encoded}", "meta": meta})
