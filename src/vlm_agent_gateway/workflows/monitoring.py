"""
Video monitoring workflow with ReAct-style structured output.

Video source ──► Frame extraction ──► VLM analysis ──► Alert dispatch
"""

import contextlib
import json
import logging
import time
from datetime import datetime, timezone

from vlm_agent_gateway.config import MONITOR_SYSTEM_PROMPT
from vlm_agent_gateway.models import AlertEvent
from vlm_agent_gateway.providers import build_video_payload, normalize_response, send_request
from vlm_agent_gateway.tools.parsing import parse_monitor_output
from vlm_agent_gateway.video import extract_frames_cv2, frames_to_base64

log = logging.getLogger("vlm-agent-gateway.monitoring")


def run_monitoring_cycle(
    endpoint: str,
    api_key: str,
    model: str,
    frame_b64_list: list[str],
    alert_prompt: str,
    max_tokens: int = 1024,
    detail: str = "low",
) -> AlertEvent:
    """
    Run one monitoring cycle: send frames → parse agent response → return AlertEvent.
    """
    user_prompt = (
        f"Monitoring condition: {alert_prompt}\n\n"
        f"You are receiving {len(frame_b64_list)} sequential video frames. "
        f"Analyze them as a temporal sequence and determine whether the "
        f"monitoring condition is present."
    )

    payload = build_video_payload(
        model=model,
        system_prompt=MONITOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        frame_b64_list=frame_b64_list,
        max_tokens=max_tokens,
        detail=detail,
    )

    t0 = time.time()
    response = send_request(api_key, endpoint, payload)
    latency_ms = (time.time() - t0) * 1000

    text = normalize_response(response)
    parsed = parse_monitor_output(text)

    return AlertEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        alert=parsed["alert"],
        summary=parsed["summary"],
        confidence=parsed["confidence"],
        recommended_action=parsed["recommended_action"],
        thought=parsed["thought"],
        latency_ms=round(latency_ms, 1),
    )


def alert_handler_console(event: AlertEvent) -> None:
    """Print alert to console with color coding."""
    prefix = "\033[91m🚨 ALERT\033[0m" if event.alert else "\033[92m✅ OK\033[0m"

    print(f"\n{prefix}  [{event.timestamp}]  latency={event.latency_ms:.0f}ms")
    print(f"  Summary: {event.summary}")
    if event.alert:
        print(f"  Confidence: {event.confidence}")
        print(f"  Action: {event.recommended_action}")
    print()


def alert_handler_jsonl(event: AlertEvent, path: str = "alerts.jsonl") -> None:
    """Append alert event as a JSON line to a file."""
    with open(path, "a") as f:
        f.write(json.dumps(event.__dict__) + "\n")


def run_continuous_monitoring(
    source,
    endpoint: str,
    api_key: str,
    model: str,
    alert_prompt: str,
    fps: float = 0.5,
    window_frames: int = 8,
    interval_seconds: float = 10.0,
    max_tokens: int = 1024,
    detail: str = "low",
    output_jsonl: str | None = None,
) -> None:
    """
    Continuously capture frames and run monitoring cycles.

    Every `interval_seconds`, captures `window_frames` frames at `fps`,
    sends them to the VLM, and processes the alert.
    """
    try:
        import cv2
    except ImportError as err:
        raise ImportError("opencv-python is required for continuous monitoring") from err

    with contextlib.suppress(ValueError, TypeError):
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    log.info(
        "Starting continuous monitoring: source=%s, model=%s, interval=%.0fs, window=%d frames @ %.1f fps",
        source,
        model,
        interval_seconds,
        window_frames,
        fps,
    )

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / fps))

    try:
        while True:
            # Collect a window of frames
            frames: list[bytes] = []
            idx = 0
            while len(frames) < window_frames:
                ret, frame = cap.read()
                if not ret:
                    # For files, loop back; for live, break
                    if isinstance(source, int):
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                if idx % frame_interval == 0:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        frames.append(buf.tobytes())
                idx += 1

            if not frames:
                log.warning("No frames captured, retrying...")
                time.sleep(interval_seconds)
                continue

            b64_frames = frames_to_base64(frames)

            try:
                event = run_monitoring_cycle(
                    endpoint=endpoint,
                    api_key=api_key,
                    model=model,
                    frame_b64_list=b64_frames,
                    alert_prompt=alert_prompt,
                    max_tokens=max_tokens,
                    detail=detail,
                )
                alert_handler_console(event)
                if output_jsonl:
                    alert_handler_jsonl(event, output_jsonl)
            except Exception as exc:
                log.error("Monitoring cycle failed: %s", exc)

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        log.info("Monitoring stopped by user")
    finally:
        cap.release()


def run_monitoring(
    video_source: str,
    endpoint: str,
    api_key: str,
    model: str,
    alert_prompt: str,
    fps: float = 1.0,
    max_frames: int = 16,
    detail: str = "low",
    max_tokens: int = 1024,
    continuous: bool = False,
    interval_seconds: float = 10.0,
    window_frames: int = 8,
    output_jsonl: str | None = None,
) -> dict:
    """
    Run video monitoring workflow.

    Single-shot mode: Extract frames, analyze, return result.
    Continuous mode: Loop with frame windows and alert handlers.
    """
    if continuous:
        # Continuous mode runs indefinitely
        run_continuous_monitoring(
            source=video_source,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            alert_prompt=alert_prompt,
            fps=fps,
            window_frames=window_frames,
            interval_seconds=interval_seconds,
            max_tokens=max_tokens,
            detail=detail,
            output_jsonl=output_jsonl,
        )
        return {"workflow": "monitor", "mode": "continuous", "status": "stopped"}

    # Single-shot mode
    log.info("Extracting frames from: %s", video_source)
    frames = extract_frames_cv2(video_source, fps=fps, max_frames=max_frames)
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_source}")

    b64_frames = frames_to_base64(frames)
    log.info("Sending %d frames to %s", len(b64_frames), model)

    event = run_monitoring_cycle(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        frame_b64_list=b64_frames,
        alert_prompt=alert_prompt,
        max_tokens=max_tokens,
        detail=detail,
    )

    alert_handler_console(event)

    if output_jsonl:
        alert_handler_jsonl(event, output_jsonl)

    return {
        "workflow": "monitor",
        "mode": "single",
        "model": model,
        "frames_analyzed": len(b64_frames),
        "alert": event.alert,
        "summary": event.summary,
        "confidence": event.confidence,
        "recommended_action": event.recommended_action,
        "thought": event.thought,
        "latency_ms": event.latency_ms,
        "timestamp": event.timestamp,
    }
