"""Backend-aware int8 dynamic quantization.

PyTorch's quantized CPU kernels need a "QEngine" registered -- `qnnpack` on
ARM (Apple Silicon, AWS Graviton), `fbgemm` on x86. Hardcoding one breaks the
other, and the serving container may end up on either depending on which EC2
instance family it's deployed to, so this picks whatever the current machine
actually supports rather than assuming.
"""
import torch


def quantize_for_serving(model: torch.nn.Module) -> tuple[torch.nn.Module, str | None]:
    """Returns (model, engine_used). engine_used is None if no quantized
    backend is available on this machine, in which case `model` is the
    original fp32 model, unchanged -- serving still works, just heavier."""
    supported = torch.backends.quantized.supported_engines
    engine = next((e for e in ("fbgemm", "qnnpack") if e in supported), None)
    if engine is None:
        return model, None
    torch.backends.quantized.engine = engine
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized, engine
