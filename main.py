from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2

from face_verifier import FaceVerifier

app = FastAPI()

verifier = FaceVerifier()

app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.post("/register")
async def register(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    success = verifier.register_reference(frame)

    return {"registered": success}


@app.post("/frame")
async def process_frame(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    result = verifier.analyze_frame(frame)

    return result
