import os
import shutil
import tempfile
import yt_dlp
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse

# Phase 5: YouTube Music processing
def filter_duration_and_live(info, *, incomplete):
    """Reject live streams and videos over 20 minutes to save Render CPU/resources."""
    duration = info.get('duration')
    if duration and duration > 1200:
        return 'Video is too long (max 20 minutes)'
    if info.get('is_live'):
        return 'Live streams are not supported'
    return None

def process_and_stream_ytmusic(url: str, background_tasks: BackgroundTasks):
    temp_dir = tempfile.mkdtemp()
    
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'extract_audio': True,
            'audio_format': 'mp3',
            'audio_quality': '0',
            'outtmpl': os.path.join(temp_dir, '%(artist)s - %(title)s.%(ext)s'),
            'noplaylist': True,
            'match_filter': filter_duration_and_live,
            # The default web player client requires solving YouTube's
            # signature/n-challenge, which needs an external JS runtime
            # yt-dlp doesn't ship. The android/ios clients receive
            # pre-resolved stream URLs and skip that challenge entirely.
            'extractor_args': {
                'youtube': {'player_client': ['android', 'ios']}
            },
        }

        # Optional: YT_COOKIES env var holding the full contents of a
        # Netscape-format cookies.txt exported from a logged-in browser
        # session. YouTube sometimes blocks Render's server IPs with a
        # "Sign in to confirm you're not a bot" error; passing cookies
        # from a real browser session works around that.
        cookies_content = os.getenv("YT_COOKIES")
        if cookies_content:
            cookies_path = os.path.join(temp_dir, "cookies.txt")
            with open(cookies_path, "w") as f:
                f.write(cookies_content)
            ydl_opts['cookiefile'] = cookies_path
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            
            # Predict the final mp3 filename based on the extracted extension
            base, ext = os.path.splitext(filename)
            mp3_filename = base + ".mp3"
        
        if not os.path.exists(mp3_filename):
            mp3_filename = filename # Fallback if convert failed but original exists
            
        def cleanup():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Ensures Render disk space doesn't fill up
        background_tasks.add_task(cleanup)
        
        return FileResponse(
            mp3_filename,
            media_type="audio/mpeg",
            filename=os.path.basename(mp3_filename),
            content_disposition_type="attachment"
        )
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
