import os
import shutil
import uuid
import mimetypes
import httpx
from fastapi import FastAPI, Request, File, UploadFile, Depends, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .storage import is_file, list_directory, upload_temp_file, HF_REPO_ID
from .auth import verify_token

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

CDN_BASE_URL = os.getenv("CDN_BASE_URL", "https://cdn-zt7p.onrender.com")
RAW_DOMAIN = os.getenv("RAW_DOMAIN", "raw.cdn.amit.is-a.dev")
RAW_BASE_URL = os.getenv("RAW_BASE_URL", f"https://{RAW_DOMAIN}")
RAW_PREFIX = "raw/"

async def stream_raw(path: str):
    hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{path}"
    client = httpx.AsyncClient(follow_redirects=True)
    req = client.build_request("GET", hf_url)
    r = await client.send(req, stream=True)
    
    if r.status_code != 200:
        await client.aclose()
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = path.split("/")[-1].lower()
    guessed_type, _ = mimetypes.guess_type(filename)
    
    # Allow media and PDFs to render in-browser securely
    if guessed_type and (
        guessed_type.startswith("image/") or 
        guessed_type.startswith("video/") or 
        guessed_type.startswith("audio/") or
        guessed_type == "application/pdf"
    ):
        content_type = guessed_type
    else:
        # Force all other files (code, HTML, JSON, unknown) to plain text
        # This prevents XSS attacks on the raw domain
        content_type = "text/plain; charset=utf-8"

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=31536000",
        "Content-Type": content_type,
        "Content-Disposition": "inline",
        "X-Content-Type-Options": "nosniff"
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

@app.post("/api/upload")
async def handle_upload(files: list[UploadFile] = File(...), _ = Depends(verify_token)):
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

@app.api_route("/ping", methods=["GET", "HEAD"], response_class=PlainTextResponse)
async def ping():
    return "Server is awake!"

@app.api_route("/{path:path}", methods=["GET", "HEAD"])
async def serve(request: Request, path: str):
    clean_path = path.strip("/")

    # 1. Intercept requests coming to the raw subdomain
    if request.url.hostname == RAW_DOMAIN:
        if not clean_path:
            return HTMLResponse("Specify a file path.", status_code=200)
        return await stream_raw(clean_path)

    # 2. Keep old /raw/ prefix working on main domain by redirecting to subdomain
    if clean_path == RAW_PREFIX.rstrip("/") or clean_path.startswith(RAW_PREFIX):
        raw_path = clean_path[len(RAW_PREFIX):]
        if not raw_path:
            return HTMLResponse("/raw/ — specify a file path after this prefix.", status_code=200)
        return RedirectResponse(f"{RAW_BASE_URL}/{raw_path}")

    if clean_path == "upload":
        return templates.TemplateResponse(request, "index.html", {"page": "upload"})
    
    if clean_path and await is_file(clean_path):
        filename = clean_path.split("/")[-1]
        return templates.TemplateResponse(request, "index.html", {
            "page": "file", 
            "path": clean_path,
            "filename": filename,
            "raw_base_url": RAW_BASE_URL
        })
    
    items = list_directory(clean_path)
    if clean_path and not items:
        raise HTTPException(status_code=404, detail="Not Found")

    return templates.TemplateResponse(request, "index.html", {
        "page": "listing",
        "path": clean_path,
        "items": items,
        "raw_base_url": RAW_BASE_URL
    })
