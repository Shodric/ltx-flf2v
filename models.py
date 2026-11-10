"""Find the four LTX-2.5 files and symlink them into ComfyUI.

This module does not download. A missing file is an error. The enhancer
checkpoint is never required and never linked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from workflow import AUDIO_VAE_NAME, CLIP_NAME, ENHANCER_FILENAME, UNET_NAME, VIDEO_VAE_NAME

REPO_DIR = "models--Lightricks--LTX-2.5"

# Comfy folder under models/, then filename.
WEIGHTS: dict[str, tuple[str, str]] = {
    "unet": ("diffusion_models", UNET_NAME),
    "clip": ("text_encoders", CLIP_NAME),
    "video_vae": ("vae", VIDEO_VAE_NAME),
    "audio_vae": ("vae", AUDIO_VAE_NAME),
}

_KNOWN_SUBDIRS = (
    "diffusion_models",
    "text_encoders",
    "vae",
    "unet",
    "clip",
    "models/diffusion_models",
    "models/text_encoders",
    "models/vae",
)


class ModelMissing(FileNotFoundError):
    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("missing weights: " + ", ".join(self.missing))


@dataclass(frozen=True)
class WeightSet:
    files: dict[str, Path]

    @property
    def ok(self) -> bool:
        return len(self.files) == len(WEIGHTS)


def default_roots() -> list[Path]:
    """COMFY_MODEL_DIR, then the Runpod HF snapshot, then a network volume."""
    roots: list[Path] = []
    override = os.environ.get("COMFY_MODEL_DIR")
    if override:
        roots.append(Path(override))
    for parent in (Path("/runpod-volume"), Path("/workspace")):
        roots.extend(_snapshot_dirs(parent / "huggingface-cache" / "hub" / REPO_DIR))
        roots.append(parent / "models")
        roots.append(parent / "ComfyUI" / "models")
    return roots


def resolve_weights(roots: list[Path]) -> WeightSet:
    found: dict[str, Path] = {}
    missing: list[str] = []
    for key, (_folder, filename) in WEIGHTS.items():
        if filename == ENHANCER_FILENAME:
            raise RuntimeError("enhancer file must not be a required weight")
        path = _find(roots, filename)
        if path is None:
            missing.append(filename)
        else:
            found[key] = path
    if missing:
        raise ModelMissing(missing)
    return WeightSet(files=found)


def link_into(comfy_dir: Path, weights: WeightSet) -> None:
    """Symlink resolved files into Comfy's model folders. Never copies."""
    for key, (folder, filename) in WEIGHTS.items():
        source = weights.files[key]
        dest_dir = comfy_dir / "models" / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        if dest.is_symlink() or dest.exists():
            if dest.resolve() == source.resolve():
                continue
            if dest.is_symlink():
                dest.unlink()
            else:
                continue
        if not dest.exists():
            dest.symlink_to(source)


def _snapshot_dirs(repo: Path) -> list[Path]:
    if not repo.is_dir():
        return []
    ref = repo / "refs" / "main"
    if ref.is_file():
        snap = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snap.is_dir():
            return [snap]
    snap_root = repo / "snapshots"
    if not snap_root.is_dir():
        return []
    return sorted(path for path in snap_root.iterdir() if path.is_dir())


def _find(roots: list[Path], filename: str) -> Path | None:
    for root in roots:
        found = _find_in(root, filename)
        if found is not None:
            return found
    return None


def _find_in(root: Path, filename: str) -> Path | None:
    if not root.is_dir():
        return None
    direct = root / filename
    if direct.is_file():
        return direct
    for sub in _KNOWN_SUBDIRS:
        candidate = root / sub / filename
        if candidate.is_file():
            return candidate
    return None
