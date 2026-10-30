"""onnxruntime sessions on the best device this build has.

``OPENCINE_CV_DEVICE=auto`` (default): CUDA with ``onnxruntime-gpu``, else DirectML with
``onnxruntime-directml`` (the ``vision``/``mask`` extras on Windows; ~4–7× faster than the CPU
for SAM 2.1 and EdgeTAM), else the CPU; ``cpu`` forces the CPU. A GPU session that fails to
build falls back to the CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from opencine.core.config import get_settings

log = logging.getLogger(__name__)

CPU = "CPUExecutionProvider"
DML = "DmlExecutionProvider"
CUDA = "CUDAExecutionProvider"


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def providers(gpu: bool = True) -> list[Any]:
    """Execution providers for a session, best first; ``gpu=False`` = CPU only."""
    import onnxruntime as ort

    have = ort.get_available_providers()
    if gpu and get_settings().cv_device != "cpu":
        if CUDA in have:
            return [(CUDA, {"cudnn_conv_algo_search": "HEURISTIC"}), CPU]
        if DML in have:  # the discrete GPU, not the first adapter (often an iGPU)
            return [(DML, {"performance_preference": "high_performance", "device_filter": "gpu"}), CPU]
    return [CPU]


def session(path: Path, gpu: bool = True) -> Any:
    """An ``InferenceSession`` for ``path`` on the GPU when there is one (``gpu=False``: CPU)."""
    import onnxruntime as ort

    eps = providers(gpu)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if eps[0][0] == DML:  # DirectML wants these off
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        sess = ort.InferenceSession(str(path), sess_options=opts, providers=eps)
    except Exception as e:  # noqa: BLE001 — a GPU provider that cannot load: CPU
        if eps == [CPU]:
            raise
        log.warning("%s on %s failed (%s); using the CPU", path.name, eps[0][0], e)
        return ort.InferenceSession(str(path), providers=[CPU])
    if eps != [CPU] and device(sess) == "cpu":  # onnxruntime fell back by itself
        log.warning("%s: %s did not start; running on the CPU", path.name, eps[0][0])
    return sess


def device(sess: Any) -> str:
    """``gpu`` or ``cpu``: where a session actually runs."""
    return "gpu" if sess.get_providers()[0] in (DML, CUDA) else "cpu"
