import json
import random
import uuid
import time
from typing import Union

from pydantic import BaseModel
from fastapi import FastAPI, Request, BackgroundTasks, UploadFile



from clipper import Clipper

app = FastAPI()


def process_clip(_id, data):
    """Process clip in the background"""
    clipper = Clipper(None, data, None)
    result = clipper.run()

    # Save result to /tmp{_id}.json
    with open(f"/tmp/{_id}.json", "w") as f:
        json.dump(result, f)

    print("Done processing clip")

def cut_audio(_id, result, audio):
    """Cut audio from result"""
    save_loc = f"/tmp/{_id}_clip.wav"
    clipper = Clipper(audio, None, save_loc=save_loc)

    clipper.cut_audio(result)
    return save_loc


@app.post("/clip")
async def clip(request:Request, background_tasks: BackgroundTasks):
    """Upload audio transcript to kick off processing"""
    body = await request.body()
    data = json.loads(body)
    
    _id = str(uuid.uuid4())

    # Processes  processing in the background
    background_tasks.add_task(process_clip, _id=_id,  data=data)

    return {"_id": _id}

@app.post("/clip/upload_audio/{_id}")
async def upload_file(_id: str, file:UploadFile):
    """Upload audio file if required"""

    contents = await file.read()
    with open(f"/tmp/{_id}.wav", "wb") as f:
        f.write(contents)

    return {"_id": _id}

@app.get("/clip/get_text/{_id}")
async def get_text(_id: str):
    """Get text from processed clip"""

    try:
        with open(f"/tmp/{_id}.json", "r") as f:
            result = json.load(f)
    except FileNotFoundError:
        return {"error": "not found"}

    return {'text': result['text'], 'window_start_token': result['window_start_token'], 'window_end_token': result['window_end_token']}

@app.get("/clip/get_audio/{_id}")
async def get_audio(_id: str):
    """Get audio from processed clip"""
    try:
        with open(f"/tmp/{_id}.wav", "rb") as f:
            audio = f.read()
    except FileNotFoundError:
        return {"error": "not audio found"}

    try:
        with open(f"/tmp/{_id}.json", "r") as f:
            result = json.load(f)
    except FileNotFoundError:
        return {"error": "not result found"}

    save_loc = self.cut_audio(_id, result, audio)

    return {'audio_loc':save_loc}
