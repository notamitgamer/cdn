import os
from fastapi import Header, HTTPException

# Phase 6: Security
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

def verify_token(authorization: str = Header(None)):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="Server missing ADMIN_TOKEN configuration")
    
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    token = authorization.split(" ")[1]
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")