from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2

from vision_engine import VisionEngine
from face_verifier import FaceVerifier
from audio_monitor import AudioMonitor
from session_manager import SessionManager

app = FastAPI()

# --- Initialize Engines ---
vision = VisionEngine()
verifier = FaceVerifier()
audio = AudioMonitor()
session = SessionManager(vision, verifier, audio)

# --- Serve demo frontend ---
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# -----------------------------
# Register Reference Face
# -----------------------------
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

    return {
        "registered": success
    }


# -----------------------------
# Process Video Frame
# -----------------------------
@app.post("/frame")
async def process_frame(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    result = session.process_frame(frame)

    return result


# -----------------------------
# Process Audio Chunk
# -----------------------------
@app.post("/audio")
async def process_audio(audio_data: list):
    try:
        audio_array = np.array(audio_data, dtype=np.float32)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio data")

    result = session.process_audio(audio_array)

    return result
