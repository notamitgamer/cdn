import os
import shutil
import tempfile
import yt_dlp
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse

# Phase 5: YouTube Music processing
def process_and_stream_ytmusic(url: str, background_tasks: BackgroundTasks):
    temp_dir = tempfile.mkdtemp()
    
    try:
        ydl_opts = {
            'format': 'bestaudio',
            'extract_audio': True,
            'audio_format': 'mp3',
            'audio_quality': '0',
            'outtmpl': os.path.join(temp_dir, '%(artist)s - %(title)s.%(ext)s'),
            'noplaylist': True,
        }
        
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