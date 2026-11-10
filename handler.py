"""Runpod queue handler for one-pass LTX-2.5 FLF2V.

``action: validate`` returns the pinned contract and does not start Comfy.
The generate path is the only path that downloads stills or talks to Comfy.
Weights are never downloaded here. ``runpod`` is imported only under ``__main__``.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import models
import workflow
from workflow import WorkerError

COMFY_DIR = Path(os.environ.get("COMFY_DIR", "/opt/ComfyUI"))
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
MAX_STILL_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 12 * 1024 * 1024
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv")

_comfy_proc: subprocess.Popen[bytes] | None = None


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input") if isinstance(job, dict) else None
    if not isinstance(payload, dict):
        raise WorkerError("job.input must be an object")
    action = payload.get("action", "generate")
    if action == "validate":
        return workflow.contract()
    if action != "generate":
        raise WorkerError("action must be validate or generate")
    return _generate(payload)


def redact(text: str) -> str:
    token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or ""
    if token:
        text = text.replace(token, "[redacted]")
    return text


def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    request = workflow.parse_job(payload)
    weights = models.resolve_weights(models.default_roots())
    os.environ["HF_HUB_OFFLINE"] = "1"
    models.link_into(COMFY_DIR, weights)

    job_id = uuid.uuid4().hex[:12]
    input_dir = COMFY_DIR / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    start_path = input_dir / f"{job_id}_start.png"
    end_path = input_dir / f"{job_id}_end.png"
    try:
        _download_still(request.start_image_url, start_path)
        _download_still(request.end_image_url, end_path)
        prompt = workflow.build_prompt(request, start_path.name, end_path.name)
        _ensure_comfy()
        prompt_id = _queue_prompt(prompt)
        entry = _wait_history(prompt_id)
        video = _first_video(entry)
        data = _fetch_view(video)
        _delete_output(video)
        return {
            "video_base64": base64.b64encode(data).decode("ascii"),
            "content_type": "video/mp4",
            "filename": video["filename"],
            "width": request.width,
            "height": request.height,
            "frames": request.frames,
            "fps": request.fps,
            "duration_seconds": request.duration_seconds,
            "seed": request.seed,
            "audio": True,
            "audio_probed": False,
        }
    finally:
        start_path.unlink(missing_ok=True)
        end_path.unlink(missing_ok=True)


def _download_still(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ramifica-ltx-flf2v"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_STILL_BYTES:
                    raise WorkerError("still image is larger than 15 MB")
                chunks.append(block)
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError(f"could not download still: {redact(str(exc))}") from exc
    data = b"".join(chunks)
    if not data:
        raise WorkerError("still image was empty")
    dest.write_bytes(data)


def _ensure_comfy() -> None:
    global _comfy_proc
    if _ready():
        return
    main = COMFY_DIR / "main.py"
    if not main.is_file():
        raise WorkerError(f"ComfyUI is not at {main}")
    _comfy_proc = subprocess.Popen(
        ["python3", "main.py", "--listen", COMFY_HOST, "--port", str(COMFY_PORT)],
        cwd=COMFY_DIR,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if _comfy_proc.poll() is not None:
            raise WorkerError(f"ComfyUI exited with code {_comfy_proc.returncode}")
        if _ready():
            return
        time.sleep(1)
    raise WorkerError("ComfyUI did not open port 8188")


def _ready() -> bool:
    try:
        with urllib.request.urlopen(_url("/system_stats"), timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _queue_prompt(prompt: dict[str, Any]) -> str:
    body = json.dumps({"prompt": prompt, "client_id": uuid.uuid4().hex}).encode("utf-8")
    request = urllib.request.Request(
        _url("/prompt"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise WorkerError(f"Comfy /prompt HTTP {exc.code}: {redact(detail)}") from exc
    parsed = json.loads(raw)
    if parsed.get("node_errors") or parsed.get("error"):
        raise WorkerError(f"Comfy rejected the prompt: {redact(raw)}")
    prompt_id = parsed.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise WorkerError(f"Comfy /prompt returned no prompt_id: {redact(raw)}")
    return prompt_id


def _wait_history(prompt_id: str) -> dict[str, Any]:
    timeout = int(os.environ.get("COMFY_JOB_TIMEOUT_S", "1800"))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(_url(f"/history/{prompt_id}"), timeout=30) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise WorkerError(f"Comfy /history failed: {redact(str(exc))}") from exc
        history = json.loads(raw)
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise WorkerError(f"Comfy execution failed: {redact(json.dumps(status))}")
            if status.get("completed"):
                return entry
        time.sleep(2)
    raise WorkerError("Comfy did not finish before COMFY_JOB_TIMEOUT_S")


def _first_video(entry: dict[str, Any]) -> dict[str, Any]:
    videos: list[dict[str, Any]] = []
    stills: list[str] = []
    for node_out in (entry.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in ("videos", "gifs", "images"):
            for item in node_out.get(key) or []:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                name = str(item["filename"])
                if name.lower().endswith(VIDEO_SUFFIXES):
                    videos.append(item)
                else:
                    stills.append(name)
    if not videos:
        raise WorkerError(
            "Comfy returned no video"
            + (f"; stills only: {', '.join(stills)}" if stills else "")
        )
    return videos[0]


def _fetch_view(item: dict[str, Any]) -> bytes:
    filename = str(item["filename"])
    subfolder = str(item.get("subfolder") or "")
    kind = str(item.get("type") or "output")
    query = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": kind}
    )
    with urllib.request.urlopen(_url(f"/view?{query}"), timeout=120) as response:
        data = response.read(MAX_VIDEO_BYTES + 1)
    if len(data) > MAX_VIDEO_BYTES:
        raise WorkerError("mp4 is larger than 12 MB; refusing to inline it")
    if not data:
        raise WorkerError("Comfy /view returned an empty file")
    return data


def _delete_output(item: dict[str, Any]) -> None:
    filename = Path(str(item.get("filename") or "")).name
    subfolder = str(item.get("subfolder") or "")
    if not filename or ".." in Path(subfolder).parts:
        return
    path = COMFY_DIR / "output" / subfolder / filename
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _url(path: str) -> str:
    return f"http://{COMFY_HOST}:{COMFY_PORT}{path}"


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
