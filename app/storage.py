import os
import time
import uuid
import httpx
from huggingface_hub import HfApi

HF_REPO_ID = os.getenv("HF_REPO_ID", "notamitgamer/cdn")
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

_cache = {}
CACHE_TTL = 60

def _get_cache(key):
    if key in _cache and _cache[key][0] > time.time():
        return _cache[key][1]
    return None

def _set_cache(key, value):
    _cache[key] = (time.time() + CACHE_TTL, value)

async def is_file(path: str) -> bool:
    if not path:
        return False
        
    cache_key = f"is_file_{path}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{path}"
    async with httpx.AsyncClient() as client:
        r = await client.head(hf_url)
        result = r.status_code == 200
        _set_cache(cache_key, result)
        return result

def _fetch_tree(path: str):
    return list(api.list_repo_tree(
        repo_id=HF_REPO_ID, path_in_repo=path, repo_type="dataset", expand=False
    ))

def list_directory(path: str):
    cache_key = f"list_dir_{path}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    try:
        items = _fetch_tree(path)
    except Exception as e:
        print(f"[storage] list_repo_tree failed for {path!r}: {e}")
        return []
    
    files_and_folders = []
    for item in items:
        name = item.path.split("/")[-1]
        is_dir = not hasattr(item, "size")
        size = getattr(item, "size", 0)
        
        size_str = format_size(size) if not is_dir else "-"

        files_and_folders.append({
            "name": name,
            "path": item.path,
            "is_dir": is_dir,
            "size_str": size_str
        })
    
    files_and_folders.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    
    _set_cache(cache_key, files_and_folders)
    return files_and_folders

ZIP_MAX_FILES = 300
ZIP_MAX_TOTAL_BYTES = 500 * 1024 * 1024

def list_files_recursive(path: str):
    try:
        items = list(api.list_repo_tree(
            repo_id=HF_REPO_ID, path_in_repo=path, repo_type="dataset", recursive=True
        ))
    except Exception:
        return []

    files = []
    total_size = 0
    for item in items:
        if hasattr(item, "size"):
            size = getattr(item, "size", 0) or 0
            total_size += size
            files.append({"path": item.path, "size": size})
            if len(files) > ZIP_MAX_FILES or total_size > ZIP_MAX_TOTAL_BYTES:
                raise ValueError(
                    f"Folder too large to zip (limit: {ZIP_MAX_FILES} files / "
                    f"{ZIP_MAX_TOTAL_BYTES // (1024 * 1024)} MB)."
                )
    return files

def format_size(size_bytes) -> str:
    if size_bytes is None:
        return "-"
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.2f} {units[idx]}"

REPO_STATS_CACHE_TTL = 300

def repo_stats():
    cache_key = "repo_stats"
    now = time.time()
    if cache_key in _cache and _cache[cache_key][0] > now:
        return _cache[cache_key][1]

    try:
        items = list(api.list_repo_tree(
            repo_id=HF_REPO_ID, path_in_repo="", repo_type="dataset", recursive=True
        ))
    except Exception as e:
        print(f"[storage] repo_stats failed: {e}")
        return None

    file_count = 0
    total_size = 0
    for item in items:
        if hasattr(item, "size"):
            file_count += 1
            total_size += getattr(item, "size", 0) or 0

    result = {"file_count": file_count, "size_str": format_size(total_size)}
    _cache[cache_key] = (now + REPO_STATS_CACHE_TTL, result)
    return result

def upload_temp_file(temp_path: str, filename: str) -> str:
    slug = str(uuid.uuid4())[:8]
    safe_filename = filename.replace(" ", "_")
    hf_path = f"uploads/{slug}-{safe_filename}"
    
    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo=hf_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    
    _cache.clear()
    return hf_path
