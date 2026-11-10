"""One-pass LTX-2.5 first/last-frame prompt.

Flattened from the official Comfy-Org template
``video_ltx2_5_flf2v`` (subgraph ``cf70afc4-5a03-47ce-8210-734b1de6c6bc``).
The enhancer, the spatial upscaler, and ComfyMathExpression are omitted.
Frame count and fps are literals.

ComfyUI schema pin: d9238b25d23e0354b0169c8197b63eca7b46443f.
ResizeImageMaskNode keys are the dotted prompt names from that schema
(``resize_type.width``, ``resize_type.height``, ``resize_type.crop``).
This module does not POST the graph. A GPU /prompt has not been run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

COMFY_REF = "d9238b25d23e0354b0169c8197b63eca7b46443f"

PRODUCT_WIDTH = 480
PRODUCT_HEIGHT = 864
PRODUCT_FPS = 24
PRODUCT_DURATION_SECONDS = 10
MIN_EDGE = 64

UNET_NAME = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
CLIP_NAME = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VIDEO_VAE_NAME = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE_NAME = "ltx-2.5-audio-vae-bf16.safetensors"

# Distilled schedule from the official template. Eight steps, no second pass.
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

NEGATIVE_PROMPT = "blurry, low quality, on-screen text, subtitles, narration, watermark"

GUIDE_STRENGTH = 0.7
IMG_COMPRESSION = 18
VIDEO_CFG = 1
AUDIO_CFG = 1
SAMPLER_ETA = 0
SAMPLER_S_NOISE = 1
TILE_SIZE = 512
TILE_OVERLAP = 64
TEMPORAL_SIZE = 64
TEMPORAL_OVERLAP = 16

ENHANCER_FILENAME = "gemma4_e2b_it_int8_convrot.safetensors"
UPSCALE_KEYS = ("upscale", "upscaler", "spatial_upscale", "second_pass")


class WorkerError(ValueError):
    """A job the worker refuses before Comfy starts."""


@dataclass(frozen=True)
class ClipRequest:
    prompt: str
    start_image_url: str
    end_image_url: str
    seed: int
    width: int
    height: int
    fps: int
    duration_seconds: int
    frames: int


def frame_count(duration_seconds: int, fps: int) -> int:
    frames = duration_seconds * fps + 1
    if (frames - 1) % 8 != 0:
        raise WorkerError(
            f"frame count {frames} is not 8n+1 "
            f"(duration_seconds={duration_seconds}, fps={fps})"
        )
    return frames


def contract() -> dict[str, Any]:
    frames = frame_count(PRODUCT_DURATION_SECONDS, PRODUCT_FPS)
    return {
        "ok": True,
        "action": "validate",
        "comfy_ref": COMFY_REF,
        "audio": True,
        "audio_probed": False,
        "width": PRODUCT_WIDTH,
        "height": PRODUCT_HEIGHT,
        "fps": PRODUCT_FPS,
        "duration_seconds": PRODUCT_DURATION_SECONDS,
        "frames": frames,
        "sampler": "SamplerCustomAdvanced",
        "sigmas": SIGMAS,
        "video_cfg": VIDEO_CFG,
        "audio_cfg": AUDIO_CFG,
        "upscaler": False,
        "prompt_enhance": False,
        "weights_baked": False,
        "gpu_tested": False,
        "image_built": False,
        "deployed": False,
    }


def parse_job(payload: dict[str, Any] | None) -> ClipRequest:
    if not isinstance(payload, dict):
        raise WorkerError("input must be an object")
    if "workflow" in payload:
        raise WorkerError("raw workflow override is refused")
    if _truthy(payload.get("prompt_enhance")):
        raise WorkerError("prompt_enhance is refused; the enhancer model is not loaded")
    for key in UPSCALE_KEYS:
        if _truthy(payload.get(key)):
            raise WorkerError(f"{key} is refused; this worker is one pass with no upscaler")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkerError("prompt is required")
    if len(prompt) > 4000:
        raise WorkerError("prompt is longer than 4000 characters")

    start_url = _image_url(payload.get("start_image_url"), "start_image_url")
    end_url = _image_url(payload.get("end_image_url"), "end_image_url")
    if "start_image_base64" in payload or "end_image_base64" in payload:
        raise WorkerError("send start_image_url and end_image_url; base64 images are refused")

    width = _bounded_edge(payload.get("width", PRODUCT_WIDTH), "width", PRODUCT_WIDTH)
    height = _bounded_edge(payload.get("height", PRODUCT_HEIGHT), "height", PRODUCT_HEIGHT)
    fps = _int(payload.get("fps", PRODUCT_FPS), "fps")
    duration = _int(payload.get("duration_seconds", PRODUCT_DURATION_SECONDS), "duration_seconds")
    if fps < 1 or fps > PRODUCT_FPS:
        raise WorkerError(f"fps must be from 1 to {PRODUCT_FPS}")
    if duration < 1 or duration > PRODUCT_DURATION_SECONDS:
        raise WorkerError(f"duration_seconds must be from 1 to {PRODUCT_DURATION_SECONDS}")
    frames = frame_count(duration, fps)

    seed_value = payload.get("seed")
    if seed_value is None:
        seed = random.SystemRandom().randrange(0, 2**31)
    else:
        seed = _int(seed_value, "seed")
        if seed < 0 or seed > 2**31 - 1:
            raise WorkerError("seed must be from 0 to 2147483647")

    return ClipRequest(
        prompt=prompt.strip(),
        start_image_url=start_url,
        end_image_url=end_url,
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration,
        frames=frames,
    )


def build_prompt(request: ClipRequest, start_name: str, end_name: str) -> dict[str, dict[str, Any]]:
    """API prompt. Node ids are strings. Links are [id, slot]."""
    width = request.width
    height = request.height
    fps = float(request.fps)
    frames = request.frames

    def n(class_type: str, inputs: dict[str, Any], title: str) -> dict[str, Any]:
        return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}

    return {
        "1": n("LoadImage", {"image": start_name}, "Load first frame"),
        "2": n("LoadImage", {"image": end_name}, "Load last frame"),
        "3": n(
            "ResizeImageMaskNode",
            {
                "input": ["1", 0],
                "resize_type": "scale dimensions",
                "resize_type.width": width,
                "resize_type.height": height,
                "resize_type.crop": "center",
                "scale_method": "nearest-exact",
            },
            "Resize first frame",
        ),
        "4": n(
            "ResizeImageMaskNode",
            {
                "input": ["2", 0],
                "resize_type": "scale dimensions",
                "resize_type.width": width,
                "resize_type.height": height,
                "resize_type.crop": "center",
                "scale_method": "nearest-exact",
            },
            "Resize last frame",
        ),
        "5": n(
            "LTXVPreprocess",
            {"image": ["3", 0], "img_compression": IMG_COMPRESSION},
            "Preprocess first frame",
        ),
        "6": n(
            "LTXVPreprocess",
            {"image": ["4", 0], "img_compression": IMG_COMPRESSION},
            "Preprocess last frame",
        ),
        "7": n("GetImageSize", {"image": ["3", 0]}, "Size of resized first frame"),
        "8": n(
            "EmptyLTXVLatentVideo",
            {"width": ["7", 0], "height": ["7", 1], "length": frames, "batch_size": 1},
            "Empty video latent",
        ),
        "9": n(
            "CLIPLoader",
            {"clip_name": CLIP_NAME, "type": "ltxv", "device": "default"},
            "LTX text encoder",
        ),
        "10": n(
            "CLIPTextEncode",
            {"text": request.prompt, "clip": ["9", 0]},
            "Positive prompt",
        ),
        "11": n(
            "CLIPTextEncode",
            {"text": NEGATIVE_PROMPT, "clip": ["9", 0]},
            "Negative prompt",
        ),
        "12": n(
            "LTXVConditioning",
            {"positive": ["10", 0], "negative": ["11", 0], "frame_rate": fps},
            "Conditioning",
        ),
        "13": n("VAELoader", {"vae_name": VIDEO_VAE_NAME}, "Video VAE"),
        "14": n("VAELoader", {"vae_name": AUDIO_VAE_NAME}, "Audio VAE"),
        "15": n(
            "UNETLoader",
            {"unet_name": UNET_NAME, "weight_dtype": "default"},
            "Distilled transformer",
        ),
        "16": n(
            "LTXVAddGuide",
            {
                "positive": ["12", 0],
                "negative": ["12", 1],
                "vae": ["13", 0],
                "latent": ["8", 0],
                "image": ["5", 0],
                "frame_idx": 0,
                "strength": GUIDE_STRENGTH,
            },
            "Guide first frame",
        ),
        "17": n(
            "LTXVAddGuide",
            {
                "positive": ["16", 0],
                "negative": ["16", 1],
                "vae": ["13", 0],
                "latent": ["16", 2],
                "image": ["6", 0],
                "frame_idx": -1,
                "strength": GUIDE_STRENGTH,
            },
            "Guide last frame",
        ),
        "18": n(
            "LTXVEmptyLatentAudio",
            {
                "frames_number": frames,
                "frame_rate": fps,
                "batch_size": 1,
                "audio_vae": ["14", 0],
            },
            "Empty audio latent",
        ),
        "19": n(
            "LTXVConcatAVLatent",
            {"video_latent": ["17", 2], "audio_latent": ["18", 0]},
            "Join audio and video",
        ),
        "20": n("RandomNoise", {"noise_seed": request.seed}, "Noise"),
        "21": n("ManualSigmas", {"sigmas": SIGMAS}, "Distilled sigmas"),
        "22": n(
            "SamplerEulerAncestral",
            {"eta": SAMPLER_ETA, "s_noise": SAMPLER_S_NOISE},
            "Euler ancestral",
        ),
        "23": n(
            "LTXVDualCFGGuider",
            {
                "model": ["15", 0],
                "positive": ["17", 0],
                "negative": ["17", 1],
                "video_cfg": VIDEO_CFG,
                "audio_cfg": AUDIO_CFG,
            },
            "CFG 1 / 1",
        ),
        "24": n(
            "SamplerCustomAdvanced",
            {
                "noise": ["20", 0],
                "guider": ["23", 0],
                "sampler": ["22", 0],
                "sigmas": ["21", 0],
                "latent_image": ["19", 0],
            },
            "One sampler pass",
        ),
        "25": n(
            "LTXVSeparateAVLatent",
            {"av_latent": ["24", 1]},
            "Split audio and video",
        ),
        "26": n(
            "LTXVCropGuides",
            {"positive": ["17", 0], "negative": ["17", 1], "latent": ["25", 0]},
            "Crop guides",
        ),
        "27": n(
            "VAEDecodeTiled",
            {
                "samples": ["26", 2],
                "vae": ["13", 0],
                "tile_size": TILE_SIZE,
                "overlap": TILE_OVERLAP,
                "temporal_size": TEMPORAL_SIZE,
                "temporal_overlap": TEMPORAL_OVERLAP,
            },
            "Decode video",
        ),
        "28": n(
            "LTXVAudioVAEDecode",
            {"samples": ["25", 1], "audio_vae": ["14", 0]},
            "Decode audio",
        ),
        "29": n(
            "CreateVideo",
            {"images": ["27", 0], "audio": ["28", 0], "fps": fps, "bit_depth": 8},
            "Mux audio into the video",
        ),
        "30": n(
            "SaveVideo",
            {
                "video": ["29", 0],
                "filename_prefix": "video/ltx_flf2v",
                "format": "mp4",
                "format.codec": "h264",
            },
            "Save mp4",
        ),
    }


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerError(f"{name} must be an integer")
    return value


def _bounded_edge(value: Any, name: str, ceiling: int) -> int:
    edge = _int(value, name)
    if edge < MIN_EDGE or edge > ceiling or edge % 32 != 0:
        raise WorkerError(
            f"{name} must be a multiple of 32 from {MIN_EDGE} to {ceiling}"
        )
    return edge


def _image_url(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.startswith(("https://", "http://")):
        raise WorkerError(f"{name} must be an http(s) URL")
    if len(value) > 2000 or any(ch.isspace() for ch in value):
        raise WorkerError(f"{name} is not a usable URL")
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
