FROM python:3.11-slim

# Phase 6: Includes ffmpeg required by yt-dlp to extract/convert to MP3
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# yt-dlp needs to keep up with YouTube's frequent changes — always pull
# the latest release on build instead of relying on the requirements.txt
# layer cache, which would otherwise pin whatever version was current
# the last time requirements.txt itself changed.
RUN pip install --no-cache-dir -U yt-dlp

# Copy the app structure
COPY app/ ./app/

EXPOSE 8000

# --proxy-headers lets FastAPI see real client IPs (needed for slowapi
# rate-limiting behind Render's load balancer)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]