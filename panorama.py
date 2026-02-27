import cv2
import numpy as np
import tempfile
import os

MAX_WIDTH = 2000    # downscale before stitching to keep memory reasonable
MAX_OUT_DIM = 16000 # safety cap on output canvas size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stitch_images(image_bytes_list: list) -> str | None:
    """
    Stitch a sequence of images (raw bytes) left-to-right in order.
    Consecutive images must overlap.
    Returns the path of the output JPEG, or None on failure.
    """
    images = []
    for raw in image_bytes_list:
        img = _decode(raw)
        if img is None:
            return None
        images.append(_resize_if_needed(img))

    if len(images) < 2:
        return None

    result = _stitch_sequential(images)
    if result is None:
        return None

    result = _crop_black_borders(result)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".jpg", delete=False, dir=_outputs_dir()
    )
    cv2.imwrite(tmp.name, result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Sequential stitcher — pairwise homographies, then single composite pass
# ---------------------------------------------------------------------------

def _stitch_sequential(images: list) -> np.ndarray | None:
    """
    1. Compute H[i]: maps images[i+1] → images[i]  (consecutive original pairs)
    2. Chain:  H_to_0[i] maps images[i] → images[0]
    3. Find canvas bounding box, then warp every image onto it in one pass.

    Matching only between consecutive originals (never against the growing
    composite) keeps the homographies stable and prevents size blow-up.
    """
    n = len(images)

    # Step 1 — pairwise homographies on original images
    H_pair = []
    for i in range(n - 1):
        H = _compute_homography(images[i], images[i + 1])
        if H is None:
            return None
        H_pair.append(H)           # H_pair[i]: images[i+1] → images[i]

    # Step 2 — chain: H_to_0[i] = H_pair[0] @ H_pair[1] @ … @ H_pair[i-1]
    H_to_0 = [np.eye(3, dtype=np.float64)]
    for i in range(1, n):
        H_to_0.append(H_to_0[i - 1] @ H_pair[i - 1])

    # Step 3 — find canvas bounding box
    all_corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, H_to_0[i]))

    pts = np.concatenate(all_corners)
    xmin, ymin = np.int32(pts.min(axis=0).ravel())
    xmax, ymax = np.int32(pts.max(axis=0).ravel())
    out_w, out_h = xmax - xmin, ymax - ymin

    if out_w <= 0 or out_h <= 0 or out_w > MAX_OUT_DIM or out_h > MAX_OUT_DIM:
        return None

    # Step 4 — warp every image onto the canvas
    tx, ty = -xmin, -ymin
    H_t = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        warped = cv2.warpPerspective(img, H_t @ H_to_0[i], (out_w, out_h))
        mask = warped.sum(axis=2) > 0
        canvas[mask] = warped[mask]

    return canvas


# ---------------------------------------------------------------------------
# Feature matching helpers
# ---------------------------------------------------------------------------

def _compute_homography(img1: np.ndarray, img2: np.ndarray) -> np.ndarray | None:
    """
    Return H such that img1_pt ≈ H @ img2_pt  (maps img2 → img1 space).
    Returns None if not enough good matches are found.
    """
    sift = cv2.SIFT_create(nfeatures=3000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50)
    )
    raw = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    if len(good) < 10:
        return None

    # findHomography(src, dst) → H where dst = H @ src
    # We pass img2 pts as src, img1 pts as dst  ⟹  img1_pt = H @ img2_pt
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or int(mask.sum()) < 8:
        return None

    return H


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _crop_black_borders(img: np.ndarray) -> np.ndarray:
    """Remove black rows/columns left over from perspective warping."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y : y + h, x : x + w]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _outputs_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "outputs")
    os.makedirs(out, exist_ok=True)
    return out
