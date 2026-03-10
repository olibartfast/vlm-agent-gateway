"""
Video frame extraction utilities.
"""

import base64
import contextlib
import logging

log = logging.getLogger("vlm-agent-gateway.video")

# cv2 is an optional dependency
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def extract_frames_cv2(
    source: str | int,
    fps: float = 1.0,
    max_frames: int = 0,
) -> list[bytes]:
    """
    Extract JPEG-encoded frames from a video file or device using OpenCV.

    Args:
        source: Video file path, RTSP URL, or device index (0 for webcam)
        fps: Target frame extraction rate (frames per second)
        max_frames: Maximum frames to extract (0 = all)

    Returns:
        List of JPEG-encoded frame bytes
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required: pip install opencv-python")

    # Interpret numeric string as device index
    with contextlib.suppress(ValueError, TypeError):
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / fps))

    frames: list[bytes] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(buf.tobytes())
            if max_frames and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    log.info("Extracted %d frames (fps=%.1f, interval=%d)", len(frames), fps, frame_interval)
    return frames


def frames_to_base64(frames: list[bytes]) -> list[str]:
    """Encode raw JPEG bytes to base64 strings."""
    return [base64.b64encode(f).decode("utf-8") for f in frames]
