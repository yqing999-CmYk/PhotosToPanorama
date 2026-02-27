import os
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.background import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

from panorama import stitch_images
from video import generate_video

app = FastAPI(title="PhotosToPanorama API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: all; prod: set to your frontend origin
    allow_methods=["*"],
    allow_headers=["*"],
)


def _remove(path: str):
    """Delete a temp file after the response has been sent."""
    try:
        os.remove(path)
    except OSError:
        pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/panorama")
async def create_panorama(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least 2 images.")
    if len(files) > 8:
        raise HTTPException(status_code=400, detail="Upload at most 8 images.")

    image_bytes = [await f.read() for f in files]

    # Run CPU-intensive stitching in a thread so the event loop stays free
    result = await run_in_threadpool(stitch_images, image_bytes)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Stitching failed. Ensure photos overlap and are in sequence.",
        )

    background_tasks.add_task(_remove, result)
    return FileResponse(
        result,
        media_type="image/jpeg",
        filename="panorama.jpg",
    )


@app.post("/api/video")
async def create_video(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least 2 images.")
    if len(files) > 8:
        raise HTTPException(status_code=400, detail="Upload at most 8 images.")

    image_bytes = [await f.read() for f in files]

    pano_path = await run_in_threadpool(stitch_images, image_bytes)
    if pano_path is None:
        raise HTTPException(
            status_code=422,
            detail="Stitching failed. Ensure photos overlap and are in sequence.",
        )

    video_path = await run_in_threadpool(generate_video, pano_path)

    background_tasks.add_task(_remove, pano_path)
    background_tasks.add_task(_remove, video_path)
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename="panorama_video.mp4",
    )
