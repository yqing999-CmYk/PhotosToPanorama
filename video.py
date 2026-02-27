import cv2
import numpy as np
import imageio
import tempfile
import os

FPS = 30
VIEWPORT_HEIGHT = 720   # divisible by 16 for codec compatibility
VIEWPORT_WIDTH = 1280   # divisible by 16 for codec compatibility
PAN_SPEED = 8           # pixels per frame (horizontal scroll)
HOLD_FRAMES = 60        # freeze frames at start and end (~2 sec)


def generate_video(panorama_path: str) -> str:
    pano_bgr = cv2.imread(panorama_path)
    if pano_bgr is None:
        raise RuntimeError("Could not read panorama file.")

    ph, pw = pano_bgr.shape[:2]

    # Scale panorama height to VIEWPORT_HEIGHT
    scale = VIEWPORT_HEIGHT / ph
    new_w = int(pw * scale)
    pano_bgr = cv2.resize(pano_bgr, (new_w, VIEWPORT_HEIGHT), interpolation=cv2.INTER_AREA)
    ph, pw = pano_bgr.shape[:2]

    # If panorama is narrower than viewport, pad with reflected edges
    if pw < VIEWPORT_WIDTH:
        pad = VIEWPORT_WIDTH - pw
        pano_bgr = cv2.copyMakeBorder(pano_bgr, 0, 0, 0, pad, cv2.BORDER_REFLECT)
        pw = pano_bgr.shape[1]

    # Convert BGR → RGB (imageio/ffmpeg expects RGB)
    pano = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

    max_x = pw - VIEWPORT_WIDTH  # max left-offset for the sliding window

    tmp = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False, dir=_outputs_dir()
    )
    tmp.close()

    # Use H.264 via imageio-ffmpeg for proper browser-compatible MP4.
    # -movflags +faststart puts the moov atom at the START of the file,
    # which is required for browsers to show correct duration and allow seeking.
    writer = imageio.get_writer(
        tmp.name,
        fps=FPS,
        codec="libx264",
        macro_block_size=16,
        output_params=["-movflags", "+faststart"],
    )

    def _write_frame(x: int):
        frame = np.ascontiguousarray(pano[:, x : x + VIEWPORT_WIDTH])
        writer.append_data(frame)

    # Hold at the left edge
    for _ in range(HOLD_FRAMES):
        _write_frame(0)

    # Pan left → right
    x = 0
    while x <= max_x:
        _write_frame(x)
        x += PAN_SPEED

    # Hold at the right edge
    for _ in range(HOLD_FRAMES):
        _write_frame(max_x)

    writer.close()
    return tmp.name


def _outputs_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "outputs")
    os.makedirs(out, exist_ok=True)
    return out
