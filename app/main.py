import os
import shutil
import uuid
import mimetypes
import io
import zipfile
import httpx
from pathlib import Path
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates

# Added format_size to the import list
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

async def stream_raw(path: str):
    hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{path}"
    client = httpx.AsyncClient(follow_redirects=True)
    req = client.build_request("GET", hf_url)
    r = await client.send(req, stream=True)
    
    if r.status_code != 200:
        await client.aclose()
        raise HTTPException(status_code=404, detail="File not found")
    
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=31536000",
        "X-Content-Type-Options": "nosniff"
    }
    
    # Safely proxy critical headers (like Content-Encoding for gzip handling)
    for h in ["Content-Type", "Content-Encoding", "Content-Length", "Etag", "Accept-Ranges"]:
        if h in r.headers:
            headers[h] = r.headers[h]
    
    # Explicit override for text files in case HF serves as octet-stream
    filename = path.split("/")[-1].lower()
    if filename.endswith(".md") or filename.endswith(".txt"):
        if headers.get("Content-Type", "application/octet-stream") == "application/octet-stream":
            headers["Content-Type"] = "text/plain; charset=utf-8"

    async def stream_generator():
        try:
            # Using aiter_raw guarantees we don't accidentally decompress the stream if it's encoded
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await client.aclose()
            
    return StreamingResponse(stream_generator(), headers=headers)

@app.get("/api/download/{path:path}")
async def download_file(path: str):
    hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{path}"
    client = httpx.AsyncClient(follow_redirects=True)
    req = client.build_request("GET", hf_url)
    r = await client.send(req, stream=True)
    
    if r.status_code != 200:
        await client.aclose()
        raise HTTPException(status_code=404, detail="Not found")
    
    filename = path.split("/")[-1]
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": r.headers.get("Content-Type", "application/octet-stream")
    }
    
    # Forward length and encoding to allow browser progress bars and raw file integrity
    for h in ["Content-Length", "Content-Encoding", "Etag"]:
        if h in r.headers:
            headers[h] = r.headers[h]
    
    async def stream_generator():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await client.aclose()
            
    return StreamingResponse(stream_generator(), headers=headers)

@app.post("/api/upload")
async def handle_upload(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        temp_path = f"/tmp/{uuid.uuid4()}-{file.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

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

    buffer = io.BytesIO()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{f['path']}"
                r = await client.get(hf_url)
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
        return await stream_raw(clean_path)

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
