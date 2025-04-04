# Face Verification System

## Overview
This project is a **Face Verification System** built using **FastAPI** and **InsightFace**. It allows users to upload videos, extracts frames, detects faces, and verifies them against a reference database using **Faiss** for efficient similarity search.

## Features
- **FastAPI-powered API** for handling face verification requests.
- **InsightFace model (buffalo_l)** for fast and accurate face detection.
- **Faiss index** for efficient face similarity searches.
- **Automatic video frame extraction** using OpenCV.
- **Supports reference face database** for identity matching.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- virtualenv (optional but recommended)

### Setup
```sh
# Clone the repository
git clone https://github.com/your-repo/FaceVerification.git
cd FaceVerification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Running the API Server
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`.

### API Endpoints
#### 1. Verify a Video
- **Endpoint:** `POST /verify-video/`
- **Description:** Upload a video to verify faces.
- **Usage:**
```sh
curl -X POST "http://localhost:8000/verify-video/" -F "file=@path/to/video.mp4"
```

## Project Structure
```
FaceVerification/
│── reference_faces/       # Folder containing reference images
│── videos/                # Temporary folder for uploaded videos
│── app.py                 # Main FastAPI app
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## Technologies Used
- **FastAPI** - API framework
- **InsightFace** - Face detection and recognition
- **Faiss** - Efficient similarity search
- **OpenCV** - Image and video processing
- **Pillow (PIL)** - Image handling

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.
