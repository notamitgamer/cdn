import os
import shutil
import httpx
from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .storage import is_file, list_directory, upload_temp_file, HF_REPO_ID
from .ytmusic import process_and_stream_ytmusic
from .auth import verify_token

# Phase 6: Rate Limiting setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory="app/templates")

# Base URL for building shareable links. Set CDN_BASE_URL on Render once
# the custom domain (cdn.amit.is-a.dev) is attached; falls back to the
# temporary onrender.com URL so this still works before that's set up.
CDN_BASE_URL = os.getenv("CDN_BASE_URL", "https://cdn-zt7p.onrender.com")

# Phase 7: Path-based raw routing (Render free tier only allows one
# custom domain, so raw serving lives at /raw/<path> instead of a
# raw. subdomain)
RAW_PREFIX = "raw/"

async def stream_raw(path: str):
    """Streams file bytes for /raw/<path> requests."""
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
        "Content-Type": r.headers.get("Content-Type", "application/octet-stream")
    }
    
    async def stream_generator():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await client.aclose()
            
    return StreamingResponse(stream_generator(), headers=headers)

@app.get("/api/download/{path:path}")
async def download_file(path: str):
    """Forces a file download from the cdn. domain."""
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
    
    async def stream_generator():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await client.aclose()
            
    return StreamingResponse(stream_generator(), headers=headers)

# Phase 4: Token-protected Upload endpoint
@app.post("/api/upload")
async def handle_upload(file: UploadFile = File(...), _ = Depends(verify_token)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        hf_path = upload_temp_file(temp_path, file.filename)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return {
        "cdn_url": f"{CDN_BASE_URL}/{hf_path}",
        "raw_url": f"{CDN_BASE_URL}/{RAW_PREFIX}{hf_path}"
    }

# Phase 5: Public, rate-limited YouTube Music downloader
@app.post("/api/ytmusic")
@limiter.limit("10/hour")
async def ytmusic_endpoint(request: Request, url: str = Form(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        return process_and_stream_ytmusic(url, background_tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{path:path}")
async def serve(request: Request, path: str):
    clean_path = path.strip("/")

    # Path-based raw serving: /raw/<path>
    if clean_path == RAW_PREFIX.rstrip("/") or clean_path.startswith(RAW_PREFIX):
        raw_path = clean_path[len(RAW_PREFIX):]
        if not raw_path:
            return HTMLResponse("/raw/ — specify a file path after this prefix.", status_code=200)
        return await stream_raw(raw_path)

    if clean_path == "upload":
        return templates.TemplateResponse(request, "index.html", {"page": "upload"})
    if clean_path == "ytmusic":
        return templates.TemplateResponse(request, "index.html", {"page": "ytmusic"})
    
    # Phase 1/2: UI Rendering (File vs Directory logic)
    if clean_path and await is_file(clean_path):
        filename = clean_path.split("/")[-1]
        return templates.TemplateResponse(request, "index.html", {
            "page": "file", 
            "path": clean_path,
            "filename": filename
        })
    
    items = list_directory(clean_path)
    if clean_path and not items:
        raise HTTPException(status_code=404, detail="Not Found")

    return templates.TemplateResponse(request, "index.html", {
        "page": "listing",
        "path": clean_path,
        "items": items
    })
