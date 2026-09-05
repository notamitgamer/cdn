import os
import shutil
import time
import uuid
import mimetypes
import io
import zipfile
import asyncio
import httpx
from collections import deque
from pathlib import Path
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates

from .storage import is_file, list_directory, list_files_recursive, upload_temp_file, repo_stats, HF_REPO_ID, format_size

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

STATIC_DIR = Path(__file__).parent / "static"

_NO_STORE_FILES = {"manifest.json", "sw.js"}

@app.get("/static/{filename}")
async def static_no_cache_root(filename: str):
    if filename not in _NO_STORE_FILES:
        raise HTTPException(status_code=404)
    file_path = STATIC_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(
        file_path,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )

@app.get("/static/icons/{filename}")
async def static_icons(filename: str):
    file_path = STATIC_DIR / "icons" / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(
        file_path,
        headers={"Cache-Control": "public, max-age=86400"},
    )

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(
        STATIC_DIR / "favicon.ico",
        headers={"Cache-Control": "public, max-age=86400"},
    )

CDN_BASE_URL = os.getenv("CDN_BASE_URL", "https://cdn-zt7p.onrender.com")
RAW_DOMAIN = os.getenv("RAW_DOMAIN", "raw.cdn.amit.is-a.dev")
RAW_BASE_URL = os.getenv("RAW_BASE_URL", f"https://{RAW_DOMAIN}")
RAW_PREFIX = "raw/"

def render_context(extra: dict) -> dict:
    stats = repo_stats()
    ctx = dict(extra)
    ctx["repo_file_count"] = stats["file_count"] if stats else None
    ctx["repo_size_str"] = stats["size_str"] if stats else None
    return ctx

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    if request.url.hostname == RAW_DOMAIN:
        return PlainTextResponse("404: File Not Found", status_code=404)
    return templates.TemplateResponse(request, "index.html", render_context({"page": "404"}), status_code=404)

async def _proxy_stream(client: httpx.AsyncClient, r: httpx.Response):
    try:
        async for chunk in r.aiter_raw():
            yield chunk
    finally:
        await client.aclose()

async def stream_raw(path: str, request: Request):
    hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{path}"
    client = httpx.AsyncClient(follow_redirects=True)
    req_headers = {}
    range_header = request.headers.get("range")
    if range_header:
        req_headers["Range"] = range_header
    req = client.build_request("GET", hf_url, headers=req_headers)
    r = await client.send(req, stream=True)

    if r.status_code not in (200, 206):
        await client.aclose()
        raise HTTPException(status_code=404, detail="File not found")

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=31536000",
        "X-Content-Type-Options": "nosniff",
        "Accept-Ranges": "bytes",
    }

    for h in ["Content-Type", "Content-Encoding", "Content-Length", "Etag", "Content-Range"]:
        if h in r.headers:
            headers[h] = r.headers[h]

    # Force text/plain for .md/.txt if HF served them as octet-stream
    filename = path.split("/")[-1].lower()
    if filename.endswith(".md") or filename.endswith(".txt"):
        if headers.get("Content-Type", "application/octet-stream") == "application/octet-stream":
            headers["Content-Type"] = "text/plain; charset=utf-8"

    return StreamingResponse(_proxy_stream(client, r), status_code=r.status_code, headers=headers)

@app.get("/api/download/{path:path}")
async def download_file(path: str, request: Request):
    hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{path}"
    client = httpx.AsyncClient(follow_redirects=True)
    req_headers = {}
    range_header = request.headers.get("range")
    if range_header:
        req_headers["Range"] = range_header
    req = client.build_request("GET", hf_url, headers=req_headers)
    r = await client.send(req, stream=True)

    if r.status_code not in (200, 206):
        await client.aclose()
        raise HTTPException(status_code=404, detail="Not found")

    filename = path.split("/")[-1]
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": r.headers.get("Content-Type", "application/octet-stream"),
        "Accept-Ranges": "bytes",
    }

    # Forward length/range/encoding so browsers can resume interrupted downloads
    for h in ["Content-Length", "Content-Encoding", "Etag", "Content-Range"]:
        if h in r.headers:
            headers[h] = r.headers[h]

    return StreamingResponse(_proxy_stream(client, r), status_code=r.status_code, headers=headers)

# Per-IP upload rate limiting: since /api/upload has no auth (shared with a
# friend), cap bandwidth instead of blocking access entirely.
UPLOAD_LIMIT_PER_MINUTE = 50 * 1024 * 1024   # 50 MB/min
UPLOAD_LIMIT_PER_HOUR = 300 * 1024 * 1024    # 300 MB/hour

class _UploadRateLimiter:
    def __init__(self):
        self._usage: dict[str, deque] = {}  # ip -> deque[(timestamp, size)]

    def _prune(self, ip: str, now: float):
        dq = self._usage.get(ip)
        if not dq:
            return
        while dq and now - dq[0][0] > 3600:
            dq.popleft()

    def check_and_record(self, ip: str, size: int):
        now = time.time()
        dq = self._usage.setdefault(ip, deque())
        self._prune(ip, now)

        minute_used = sum(s for t, s in dq if now - t <= 60)
        hour_used = sum(s for t, s in dq)

        if minute_used + size > UPLOAD_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail=f"Upload rate limit exceeded: {format_size(UPLOAD_LIMIT_PER_MINUTE)}/minute. Try again shortly.",
            )
        if hour_used + size > UPLOAD_LIMIT_PER_HOUR:
            raise HTTPException(
                status_code=429,
                detail=f"Upload rate limit exceeded: {format_size(UPLOAD_LIMIT_PER_HOUR)}/hour. Try again later.",
            )

        dq.append((now, size))

_upload_limiter = _UploadRateLimiter()

@app.post("/api/upload")
async def handle_upload(request: Request, files: list[UploadFile] = File(...)):
    client_ip = request.client.host if request.client else "unknown"
    results = []
    for file in files:
        temp_path = f"/tmp/{uuid.uuid4()}-{file.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        size = os.path.getsize(temp_path)
        try:
            _upload_limiter.check_and_record(client_ip, size)
        except HTTPException:
            os.remove(temp_path)
            raise

        try:
            hf_path = upload_temp_file(temp_path, file.filename)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        results.append({
            "filename": file.filename,
            "cdn_url": f"{CDN_BASE_URL}/{hf_path}",
            "raw_url": f"{RAW_BASE_URL}/{hf_path}"
        })

    return {"files": results}

@app.get("/api/zip-stats/{path:path}")
async def zip_stats(path: str):
    clean_path = path.strip("/")
    try:
        files = list_files_recursive(clean_path)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    if not files:
        raise HTTPException(status_code=404, detail="Folder is empty or not found")

    total_size = sum(f["size"] for f in files)
    return {
        "file_count": len(files),
        "total_size": total_size,
        "size_str": format_size(total_size)
    }

@app.get("/api/download-zip/{path:path}")
async def download_zip(path: str):
    clean_path = path.strip("/")

    try:
        files = list_files_recursive(clean_path)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    if not files:
        raise HTTPException(status_code=404, detail="Folder is empty or not found")

    prefix_len = len(clean_path.rstrip("/")) + 1 if clean_path else 0
    ZIP_FETCH_CONCURRENCY = 8
    semaphore = asyncio.Semaphore(ZIP_FETCH_CONCURRENCY)

    async def fetch(client: httpx.AsyncClient, f: dict):
        hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{f['path']}"
        async with semaphore:
            r = await client.get(hf_url)
        return f, r

    buffer = io.BytesIO()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        results = await asyncio.gather(*(fetch(client, f) for f in files))
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f, r in results:
                if r.status_code != 200:
                    continue
                arcname = f["path"][prefix_len:] if prefix_len else f["path"]
                zf.writestr(arcname, r.content)

    buffer.seek(0)
    zip_filename = (clean_path.rstrip("/").split("/")[-1] if clean_path else HF_REPO_ID.split("/")[-1]) + ".zip"

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'}
    )

@app.api_route("/ping", methods=["GET", "HEAD"], response_class=PlainTextResponse)
async def ping():
    return "Server is awake!"

@app.api_route("/{path:path}", methods=["GET", "HEAD"])
async def serve(request: Request, path: str):
    clean_path = path.strip("/")

    if request.url.hostname == RAW_DOMAIN:
        if not clean_path:
            return HTMLResponse("Specify a file path.", status_code=200)
        return await stream_raw(clean_path, request)

    if clean_path == RAW_PREFIX.rstrip("/") or clean_path.startswith(RAW_PREFIX):
        raw_path = clean_path[len(RAW_PREFIX):]
        if not raw_path:
            return HTMLResponse("/raw/ — specify a file path after this prefix.", status_code=200)
        return RedirectResponse(f"{RAW_BASE_URL}/{raw_path}")

    if clean_path == "upload":
        return templates.TemplateResponse(request, "index.html", render_context({"page": "upload"}))
    
    if clean_path and await is_file(clean_path):
        filename = clean_path.split("/")[-1]
        return templates.TemplateResponse(request, "index.html", render_context({
            "page": "file", 
            "path": clean_path,
            "filename": filename,
            "raw_base_url": RAW_BASE_URL
        }))
    
    items = list_directory(clean_path)
    if clean_path and not items:
        raise HTTPException(status_code=404, detail="Not Found")

    return templates.TemplateResponse(request, "index.html", render_context({
        "page": "listing",
        "path": clean_path,
        "items": items,
        "raw_base_url": RAW_BASE_URL
    }))
