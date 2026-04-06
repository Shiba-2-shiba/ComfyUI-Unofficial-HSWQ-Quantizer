from __future__ import annotations

try:
    from ..weighted_histogram_mse import FP8E4M3Quantizer, HSWQWeightedHistogramOptimizer
    from ..weighted_histogram_mse_fast import FP8E4M3QuantizerOptimized, HSWQWeightedHistogramOptimizerFast
except Exception:
    from weighted_histogram_mse import FP8E4M3Quantizer, HSWQWeightedHistogramOptimizer
    from weighted_histogram_mse_fast import FP8E4M3QuantizerOptimized, HSWQWeightedHistogramOptimizerFast


OPTIMIZER_MODES = {
    "classic": HSWQWeightedHistogramOptimizer,
    "fast": HSWQWeightedHistogramOptimizerFast,
}

FP8_QUANTIZER_MODES = {
    "classic": FP8E4M3Quantizer,
    "fast": FP8E4M3QuantizerOptimized,
}


def create_optimizer(*, mode: str = "fast", **kwargs):
    optimizer_cls = OPTIMIZER_MODES[mode]
    return optimizer_cls(**kwargs)


def create_fp8_quantizer(*, mode: str = "fast", device: str = "cuda"):
    quantizer_cls = FP8_QUANTIZER_MODES[mode]
    return quantizer_cls(device)

