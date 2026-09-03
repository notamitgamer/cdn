import os
import uuid
import httpx
from huggingface_hub import HfApi

# Phase 1: Storage Layer
HF_REPO_ID = os.getenv("HF_REPO_ID", "notamitgamer/cdn")
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

async def is_file(path: str) -> bool:
    """Check if a path in the HF dataset is a file using a quick HEAD request."""
    if not path:
        return False
    hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{path}"
    async with httpx.AsyncClient() as client:
        r = await client.head(hf_url)
        return r.status_code == 200

def list_directory(path: str):
    """Returns list of dicts for items in directory, sorted folders first."""
    try:
        items = list(api.list_repo_tree(repo_id=HF_REPO_ID, path_in_repo=path, repo_type="dataset"))
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

        files_and_folders.append({
            "name": name,
            "path": item.path,
            "is_dir": is_dir,
            "size_str": size_str
        })
    
    files_and_folders.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    return files_and_folders

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
    return hf_path