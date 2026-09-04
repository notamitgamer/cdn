import os
import time
import uuid
import httpx
from huggingface_hub import HfApi

# Phase 1: Storage Layer
HF_REPO_ID = os.getenv("HF_REPO_ID", "notamitgamer/cdn")
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# --- SPEED OPTIMIZATION: IN-MEMORY CACHE ---
_cache = {}
CACHE_TTL = 60  # Cache folder layouts for 60 seconds to make navigation instant

def _get_cache(key):
    if key in _cache and _cache[key][0] > time.time():
        return _cache[key][1]
    return None

def _set_cache(key, value):
    _cache[key] = (time.time() + CACHE_TTL, value)
# -------------------------------------------

async def is_file(path: str) -> bool:
    """Check if a path in the HF dataset is a file using a quick HEAD request."""
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

def list_directory(path: str):
    """Returns list of dicts for items in directory, sorted folders first."""
    cache_key = f"list_dir_{path}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    try:
        items = list(api.list_repo_tree(
            repo_id=HF_REPO_ID, path_in_repo=path, repo_type="dataset", expand_info=True
        ))
    except Exception:
        return []
    
    files_and_folders = []
    for item in items:
        name = item.path.split("/")[-1]
        is_dir = not hasattr(item, "size")
        size = getattr(item, "size", 0)
        
        size_str = "-"
        if not is_dir and size is not None:
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"

        # last_commit is only populated when expand_info=True is passed to
        # list_repo_tree; we fetch it lazily below only if available, so this
        # degrades gracefully to "-" rather than failing the whole listing.
        last_modified = getattr(item, "last_commit", None)
        modified_str = "-"
        if last_modified is not None:
            date = getattr(last_modified, "date", None)
            if date is not None:
                modified_str = date.strftime("%Y-%m-%d")

        files_and_folders.append({
            "name": name,
            "path": item.path,
            "is_dir": is_dir,
            "size_str": size_str,
            "modified_str": modified_str
        })
    
    files_and_folders.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    
    _set_cache(cache_key, files_and_folders)
    return files_and_folders

# Hard cap so a zip-download request can't be used to pull an unbounded
# amount of data through the server at once.
ZIP_MAX_FILES = 300
ZIP_MAX_TOTAL_BYTES = 500 * 1024 * 1024  # 500 MB

def list_files_recursive(path: str):
    """
    Returns a flat list of every file (not folder) under `path`, for
    building a zip archive. Raises ValueError if the folder is too big
    to safely zip in one request.
    """
    try:
        items = list(api.list_repo_tree(
            repo_id=HF_REPO_ID, path_in_repo=path, repo_type="dataset", recursive=True
        ))
    except Exception:
        return []

    files = []
    total_size = 0
    for item in items:
        if hasattr(item, "size"):  # files only, skip folder entries
            size = getattr(item, "size", 0) or 0
            total_size += size
            files.append({"path": item.path, "size": size})
            if len(files) > ZIP_MAX_FILES or total_size > ZIP_MAX_TOTAL_BYTES:
                raise ValueError(
                    f"Folder too large to zip (limit: {ZIP_MAX_FILES} files / "
                    f"{ZIP_MAX_TOTAL_BYTES // (1024 * 1024)} MB)."
                )
    return files

def upload_temp_file(temp_path: str, filename: str) -> str:
    """Uploads a local file to HF and returns the relative path."""
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
    
    # Clear cache so the newly uploaded file is immediately visible in directories
    _cache.clear() 
    return hf_path
