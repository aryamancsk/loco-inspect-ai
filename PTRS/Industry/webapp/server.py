import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .auth_db import (
    authenticate_user,
    delete_session,
    get_inspection_history,
    get_user_by_token,
    init_db,
    record_inspection,
    register_user,
)
from .inspector_engine import (
    ALL_CATEGORIES,
    CATEGORY_METADATA,
    DATA_ROOT,
    ENGINE,
    OUTPUT_DIR,
)

WEBAPP_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBAPP_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True, parents=True)

app = FastAPI(
    title="MVTec LOCO AD — Industrial Visual Anomaly & Defect Inspector",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    init_db()
    # Preload default juice_bottle model in background so first inspection is instant
    try:
        ENGINE.ensure_category_model("juice_bottle")
    except Exception as exc:
        print(f"[Startup warning] Could not preload juice_bottle model: {exc}")


async def _parse_payload(request: Request) -> Dict[str, Any]:
    """Flexibly parse either JSON body or Form data without ever throwing 422 errors."""
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            data = await request.json()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    try:
        form = await request.form()
        return {k: v for k, v in form.items()}
    except Exception:
        pass
    try:
        raw = await request.body()
        if raw:
            import json
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _extract_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization") or ""
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    x_token = request.headers.get("X-Session-Token")
    if x_token:
        return x_token.strip()
    cookie_token = request.cookies.get("session_token")
    if cookie_token:
        return cookie_token.strip()
    return None


def _get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    token = _extract_token(request)
    return get_user_by_token(token)


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="session_token",
        value=token,
        max_age=30 * 24 * 3600,
        httponly=False,
        samesite="lax",
        path="/",
    )


@app.post("/api/auth/signup")
async def api_signup(request: Request):
    payload = await _parse_payload(request)
    username = str(payload.get("username") or payload.get("identifier") or "").strip()
    email = str(payload.get("email") or "").strip()
    password = str(payload.get("password") or "").strip()
    confirm_password = payload.get("confirm_password")
    full_name = str(payload.get("full_name") or payload.get("name") or "").strip()
    role = str(payload.get("role") or "Quality Inspector").strip()

    if confirm_password is not None and str(confirm_password).strip() != "":
        if password != str(confirm_password).strip():
            return JSONResponse(
                status_code=400,
                content={"ok": False, "message": "Passwords do not match. Please re-enter your password."},
            )

    ok, msg, user, token = register_user(
        username=username,
        email=email,
        password=password,
        full_name=full_name,
        role=role,
    )
    if not ok or not user or not token:
        return JSONResponse(status_code=400, content={"ok": False, "message": msg})

    resp = JSONResponse(
        status_code=200,
        content={"ok": True, "message": msg, "user": user, "token": token},
    )
    _set_auth_cookie(resp, token)
    return resp


@app.post("/api/auth/login")
async def api_login(request: Request):
    payload = await _parse_payload(request)
    identifier = str(
        payload.get("identifier")
        or payload.get("username")
        or payload.get("email")
        or ""
    ).strip()
    password = str(payload.get("password") or "").strip()

    ok, msg, user, token = authenticate_user(identifier=identifier, password=password)
    if not ok or not user or not token:
        return JSONResponse(status_code=400, content={"ok": False, "message": msg})

    resp = JSONResponse(
        status_code=200,
        content={"ok": True, "message": msg, "user": user, "token": token},
    )
    _set_auth_cookie(resp, token)
    return resp


@app.post("/api/auth/demo")
async def api_demo_login():
    ok, msg, user, token = authenticate_user("demo", "demo1234")
    if not ok or not user or not token:
        ok, msg, user, token = register_user(
            username="demo",
            email="demo@nus.edu.sg",
            password="demo1234",
            full_name="NUS Quality Inspector (Demo)",
            role="Lead Quality Engineer",
        )
    resp = JSONResponse(
        status_code=200,
        content={"ok": True, "message": "Logged into Demo Inspector account.", "user": user, "token": token},
    )
    if token:
        _set_auth_cookie(resp, token)
    return resp


@app.post("/api/auth/logout")
async def api_logout(request: Request):
    token = _extract_token(request)
    delete_session(token)
    resp = JSONResponse(status_code=200, content={"ok": True, "message": "Logged out successfully."})
    resp.delete_cookie("session_token", path="/")
    return resp


@app.get("/api/auth/me")
async def api_me(request: Request):
    user = _get_current_user(request)
    return {"ok": True, "authenticated": user is not None, "user": user}


@app.get("/api/meta")
async def api_meta():
    return {
        "ok": True,
        "device": str(ENGINE.ensure_extractor().device) if ENGINE.extractor else ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu"),
        "categories": [
            {"id": cat_id, **CATEGORY_METADATA[cat_id]}
            for cat_id in ALL_CATEGORIES
        ],
        "samples": ENGINE.list_sample_images(),
    }


@app.get("/api/sample-image")
async def api_sample_image(rel_path: str):
    clean_rel = rel_path.replace("\\", "/").lstrip("/")
    target = (DATA_ROOT / clean_rel).resolve()
    if not str(target).startswith(str(DATA_ROOT.resolve())) or not target.exists():
        return JSONResponse(status_code=404, content={"ok": False, "message": "Sample image not found."})
    return FileResponse(str(target), media_type="image/png")


@app.post("/api/inspect")
async def api_inspect(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    category: str = Form(default="auto"),
    operating_mode: str = Form(default="optimal"),
    sample_rel_path: Optional[str] = Form(default=None),
    image_base64: Optional[str] = Form(default=None),
):
    try:
        user = _get_current_user(request)
        raw_bytes: Optional[bytes] = None
        pil_img: Optional[Image.Image] = None
        filename = "uploaded_image.png"

        # If JSON payload was sent instead of multipart form
        if file is None and not sample_rel_path and not image_base64:
            payload = await _parse_payload(request)
            category = str(payload.get("category") or category or "auto")
            operating_mode = str(payload.get("operating_mode") or operating_mode or "optimal")
            sample_rel_path = payload.get("sample_rel_path")
            image_base64 = payload.get("image_base64")

        if file is not None and file.filename:
            filename = file.filename
            raw_bytes = await file.read()
            if not raw_bytes:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "message": "The uploaded image file is empty. Please select a valid image."},
                )
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes))
                pil_img.load()
                pil_img = pil_img.convert("RGB")
            except Exception as img_exc:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "message": f"Could not decode image file '{filename}': {img_exc}"},
                )
        elif sample_rel_path:
            clean_rel = str(sample_rel_path).replace("\\", "/").lstrip("/")
            target = (DATA_ROOT / clean_rel).resolve()
            if not str(target).startswith(str(DATA_ROOT.resolve())) or not target.exists():
                return JSONResponse(
                    status_code=404,
                    content={"ok": False, "message": f"Sample image '{sample_rel_path}' was not found."},
                )
            filename = target.name
            raw_bytes = target.read_bytes()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif image_base64:
            b64_str = str(image_base64)
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_str)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        else:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "message": "Please upload a product image or choose a sample image first."},
            )

        result = ENGINE.inspect_image(
            image_input=pil_img,
            raw_bytes=raw_bytes,
            category=category,
            operating_mode=operating_mode,
            filename=filename,
        )

        inspection_id = record_inspection(
            user_id=user["id"] if user else None,
            category=result["category"],
            operating_mode=result["operating_mode"],
            filename=filename,
            decision=result["decision"],
            severity=result["severity"],
            score=result["score"],
            threshold=result["threshold"],
            z_score=result["z_score"],
            hotspot_pct=result["hotspot_pct"],
            peak_region=result["peak_region"],
            explanation=result["explanation"],
        )
        result["inspection_id"] = inspection_id
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": f"Inspection encountered an unexpected error: {type(exc).__name__}: {exc}"},
        )


@app.get("/api/history")
async def api_history(request: Request):
    user = _get_current_user(request)
    history = get_inspection_history(user_id=user["id"] if user else None, limit=30)
    return {"ok": True, "history": history}


if OUTPUT_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/download-individual-report")
async def download_individual_report():
    pdf_path = (WEBAPP_DIR.parent / "INDIVIDUAL_PROJECT_REPORT.pdf").resolve()
    if not pdf_path.exists():
        pdf_path = (WEBAPP_DIR.parent.parent.parent / "INDIVIDUAL_PROJECT_REPORT.pdf").resolve()
    if pdf_path.exists():
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            filename="NUS_ISS_Individual_Project_Report_Aryaman.pdf",
        )
    return JSONResponse(status_code=404, content={"ok": False, "message": "Individual report PDF not found."})


@app.get("/download-report")
async def download_report():
    pdf_path = (WEBAPP_DIR.parent / "FINAL_PROJECT_REPORT.pdf").resolve()
    if not pdf_path.exists():
        pdf_path = (WEBAPP_DIR.parent.parent.parent / "FINAL_PROJECT_REPORT.pdf").resolve()
    if pdf_path.exists():
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            filename="NUS_ISS_Final_Project_Report_LOCO_Inspect_AI.pdf",
        )
    return JSONResponse(status_code=404, content={"ok": False, "message": "PDF report not found."})


@app.get("/", response_class=HTMLResponse)
async def index_page():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Industrial Anomaly Inspector Loading...</h1>")
