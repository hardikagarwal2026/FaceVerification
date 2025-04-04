from fastapi import FastAPI, File, UploadFile
import numpy as np
import faiss
import os
import torch
import cv2
import random
import shutil
from PIL import Image
from insightface.app import FaceAnalysis

app = FastAPI()

# Load RetinaFace model for fast face detection
face_analyzer = FaceAnalysis(name="buffalo_l")
face_analyzer.prepare(ctx_id=-1)  # Use CPU

# Reference folder and index file
REFERENCE_FOLDER = "reference_faces"
VIDEO_FOLDER = "videos"
INDEX_FILE = "face_index.bin"

# Create video folder if not exists
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Precompute embeddings for known faces
face_db = []
face_names = []

def preprocess_image(img_path):
    """Extract face embeddings from an image"""
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    faces = face_analyzer.get(img)

    if len(faces) == 0:
        return None

    return faces[0].normed_embedding  # Return face embedding

# Load known faces
for filename in os.listdir(REFERENCE_FOLDER):
    file_path = os.path.join(REFERENCE_FOLDER, filename)
    embedding = preprocess_image(file_path)
    if embedding is not None:
        face_db.append(embedding)
        face_names.append(filename)

face_db = np.array(face_db, dtype=np.float32)
index = faiss.IndexFlatL2(face_db.shape[1])
index.add(face_db)

# Save index
faiss.write_index(index, INDEX_FILE)

def extract_random_frames(video_path):
    """Extract two random frames from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 2:
        return []  # Not enough frames to extract

    # Select two random frame indices
    frame_indices = random.sample(range(total_frames), 2)
    frame_list = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i in frame_indices:
            frame_list.append(frame)

    cap.release()
    return frame_list

@app.post("/verify-video/")
async def verify_video(file: UploadFile = File(...)):
    """Upload and process a video to verify face match."""
    video_path = os.path.join(VIDEO_FOLDER, file.filename)

    # Stream file to avoid memory issues with large files
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract two random frames
    frames = extract_random_frames(video_path)

    if not frames:
        return {"status": "error", "message": "⚠️ Not enough frames to analyze"}

    for frame in frames:
        faces = face_analyzer.get(frame)

        for face in faces:
            embedding = face.normed_embedding
            D, I = index.search(np.array([embedding], dtype=np.float32), 1)

            if D[0][0] < 0.6:  # If match found
                matched_person = face_names[I[0][0]]
                return {"status": "success", "message": f"✅ Person Found: {matched_person}"}

    return {"status": "error", "message": "❌ No matching person found in video"}
