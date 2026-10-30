"""Model files for ``opencine.vision``, pinned to a Hugging Face commit and fetched once into
``OPENCINE_MODELS_DIR/<name>`` (``OPENCINE_VISION_MODELS=false``: never; callers fall back).

- ``florence2``: Florence-2-base-ft (MIT), onnx-community export, int8 decoder (~550 MB);
- ``edgetam``: EdgeTAM video graphs (Apache-2.0, 65 MB) plus two learned tensors the export
  leaves out, range-read from the transformers checkpoint into ``extras.npz``;
- ``depth``: Depth Anything V2 Small (Apache-2.0; the larger sizes are non-commercial), 99 MB;
- ``yunet``: OpenCV's YuNet face detector (MIT, 230 KB), run by ``cv2.FaceDetectorYN``.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any

import numpy as np

from opencine.core.config import get_settings
from opencine.vision import runtime

log = logging.getLogger(__name__)

MODELS: dict[str, dict[str, Any]] = {
    "florence2": {"repo": "onnx-community/Florence-2-base-ft", "rev": "e88a44eaf3791a35eae0c5a47b3dbcd36e67eb6f",
                  "files": ["tokenizer.json", "onnx/vision_encoder.onnx", "onnx/encoder_model.onnx",
                            "onnx/decoder_model_merged_quantized.onnx", "onnx/embed_tokens_quantized.onnx"]},
    "edgetam": {"repo": "jax-image-tools/edgetam-video-onnx", "rev": "8ca3d3e4169938e65b552cdf14542fde25e8badb",
                "files": ["vision_encoder.onnx", "mask_decoder.onnx", "memory_attention.onnx",
                          "memory_encoder.onnx"]},
    "depth": {"repo": "onnx-community/depth-anything-v2-small", "rev": "4472b7362082ad9968fee890ca0f1e5aca36b93d",
              "files": ["onnx/model.onnx"]},
    "yunet": {"repo": "opencv/face_detection_yunet", "rev": "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
              "files": ["face_detection_yunet_2023mar.onnx"], "onnxruntime": False},
}
EDGETAM_EXTRAS = {"repo": "yonigozlan/EdgeTAM-hf", "rev": "c266ce53b3fc00f0f495b583f6a116c4e57f53bb",
                  "file": "model.safetensors",
                  "tensors": ("no_memory_embedding", "memory_temporal_positional_encoding")}


def fetch(name: str) -> Path | None:
    """The folder holding model ``name``'s files (downloaded on first use), or None when vision
    models are off, onnxruntime is not installed, or the download fails."""
    s = get_settings()
    spec = MODELS[name]
    if not s.vision_models or (spec.get("onnxruntime", True) and not runtime.available()):
        return None
    d = s.models_dir / name
    if _complete(name, d):
        return d
    try:
        from huggingface_hub import hf_hub_download

        for f in spec["files"]:
            if not (d / f).is_file():
                log.info("fetching %s/%s", spec["repo"], f)
                hf_hub_download(spec["repo"], f, revision=spec["rev"], local_dir=d)
        if name == "edgetam" and not (d / "extras.npz").is_file():
            e = EDGETAM_EXTRAS
            url = f"https://huggingface.co/{e['repo']}/resolve/{e['rev']}/{e['file']}"
            np.savez(d / "extras.npz", **safetensors_pick(url, e["tensors"]))
    except Exception as e:  # noqa: BLE001 — offline, HF down, disk full: the callers fall back
        log.warning("could not fetch the %s model: %s", name, e)
        return None
    return d if _complete(name, d) else None


def _complete(name: str, d: Path) -> bool:
    return all((d / f).is_file() for f in MODELS[name]["files"]) and (
        name != "edgetam" or (d / "extras.npz").is_file())


def safetensors_pick(url: str, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Tensors ``names`` (float32) of a remote ``.safetensors`` file, without downloading it."""
    import httpx

    def part(c: httpx.Client, a: int, b: int) -> bytes:  # bytes a … b-1
        r = c.get(url, headers={"Range": f"bytes={a}-{b - 1}"})
        r.raise_for_status()
        return r.content if r.status_code == 206 else r.content[a:b]

    with httpx.Client(follow_redirects=True, timeout=60) as c:
        n = struct.unpack("<Q", part(c, 0, 8))[0]
        head = json.loads(part(c, 8, 8 + n))
        out = {}
        for k in names:
            if head[k]["dtype"] != "F32":
                raise ValueError(f"{k}: expected F32, got {head[k]['dtype']}")
            a, b = head[k]["data_offsets"]
            out[k] = np.frombuffer(part(c, 8 + n + a, 8 + n + b), "<f4").reshape(head[k]["shape"])
    return out
