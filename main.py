import json
import random
import uuid
import time

from fastapi import FastAPI, Request, BackgroundTasks, UploadFile
from pydantic import BaseModel
from typing import Union

from clipper import Clipper

app = FastAPI()


def process_clip(_id, data):

    clipper = Clipper(None, data, None)
    result = clipper.run()

    # Save result to /tmp{_id}.json
    with open(f"/tmp/{_id}.json", "w") as f:
        json.dump(result, f)

    print("Done processing clip")






@app.post("/clip")
async def clip(request:Request, background_tasks: BackgroundTasks):


    body = await request.body()
    data = json.loads(body)
    
    _id = str(uuid.uuid4())

    background_tasks.add_task(process_clip, _id=_id,  data=data)

    return {"_id": _id}

@app.post("/clip/upload_audio/{_id}")
async def upload_file(_id: str, file:UploadFile):

    contents = await file.read()

    # Save file to /tmp{_id}.wav
    with open(f"/tmp/{_id}.wav", "wb") as f:
        f.write(contents)

    return {"_id": _id}

@app.get("/clip/get_text/{_id}")
async def get_text(_id: str):
    try:
        with open(f"/tmp/{_id}.json", "r") as f:
            result = json.load(f)
    except FileNotFoundError:
        return {"error": "not found"}

    return result

@app.get("/clip/get_audio/{_id}")
async def get_audio(_id: str):
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

    save_loc = f"/tmp/{_id}_clip.wav"
    clipper = Clipper(audio, None, save_loc=save_loc)

    clipper.cut_audio(result['window_start_token'], result['window_end_token'], result['item_list'])

    return {'audio_loc':save_loc}
