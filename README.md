# ComfyUI-Unofficial-HSWQ-QuantizerHSWQ-Quantizer
### Unofficial ComfyUI reference implementation of Hybrid Sensitivity Weighted Quantization (HSWQ)

## Overview
This repository provides an **unofficial reference implementation** of **Hybrid Sensitivity Weighted Quantization (HSWQ)** for ComfyUI.

The original HSWQ method and core algorithm were proposed and released by:
👉 [https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

> **Note:** This project does not modify the original algorithm. Its purpose is to make HSWQ practically usable inside ComfyUI workflows.

It provides:
* **A calibration node:** Collects HSWQ statistics during normal image generation.
* **A conversion node:** Applies V1-compatible FP8 quantization using the collected statistics.

This implementation is intended as a practical integration / reference, not as an alternative or competing implementation.

---

## What This Implementation Adds
Compared to the original scripts, this repository focuses on **workflow-level integration**:

### ComfyUI Custom Nodes
* **Calibration (statistics collection):** Hooks into the generation process.
* **FP8 conversion:** Converts models directly within ComfyUI.

### Session-Aware Calibration
* **Accumulation:** Statistics can be accumulated across multiple runs.
* **Safe Saving:** Uses atomic saving to avoid corrupted stats files.

### V1-Compatible FP8 Conversion
* **Smart Layer Selection:** Keeps top-k sensitive layers in **FP16**.
* **Optimization:** Applies weighted histogram MSE optimization for FP8 `amax` selection.

All algorithmic decisions follow the design described in the original repository.

---

## Scope and Non-Goals

### ✅ In Scope
* Practical ComfyUI integration
* Reference implementation for real workflows
* Faithful reproduction of HSWQ V1 behavior

### ❌ Out of Scope
* Proposing new quantization algorithms
* Changing HSWQ theory or selection criteria
* Replacing the original implementation

---

## Installation
Clone this repository into your ComfyUI `custom_nodes` directory:

```
cd ComfyUI/custom_nodes
git clone [https://github.com/](https://github.com/)<yourname>/ComfyUI-HSWQ-Quantizer
```

> Please restart ComfyUI after installation.

### Dependencies
This project relies on **PyTorch with FP8 support** and includes optional dependencies for the benchmark node:

* **Required:** `torch` with `torch.float8_e4m3fn` support
* **Benchmark node:** `lpips`, `open_clip_torch`

Install the benchmark dependencies if you plan to use the benchmark node:

```
pip install lpips open_clip_torch
```

## Provided Nodes

### 1. HSWQ Calibration (Dual Monitor)
Collects calibration statistics while running standard SDXL generation.

**Key features:**
* Hooks into UNet forward passes
* **Tracks:**
  * Output sensitivity (variance)
  * Input channel importance
* Session-based accumulation
* Automatic checkpointing

**Typical usage:**
1. Insert the calibration node into your SDXL workflow.
2. Run generation multiple times.
3. Statistics are saved automatically as `.pt` files.

### 2. HSWQ FP8 Converter
Converts an SDXL UNet model to FP8 using collected calibration statistics.

**Conversion process:**
1. Load calibration statistics.
2. Rank layers by sensitivity.
3. Keep top `keep_ratio` layers in **FP16**.
4. Quantize remaining layers to **FP8** (`torch.float8_e4m3fn`).
5. Optimize `amax` using weighted histogram MSE (HSWQ V1).

The output model remains compatible with standard ComfyUI loaders.

### 3. HSWQ FP8 Converter (Legacy)
The former `SDXLHSWQQuantizer.py` implementation has been moved to
`SDXLHSWQQuantizerLegacy.py`, while the latest implementation now lives in
`SDXLHSWQQuantizer.py`. Use the legacy node for compatibility comparisons with
the earlier behavior.

### 4. HSWQ Advanced Benchmark
Provides a benchmark node for comparing output fidelity across FP8/FP16 models.
This node requires the **`lpips`** and **`open_clip_torch`** packages.

---

## Recommended Settings
These settings follow the guidance from the original HSWQ repository:

| Parameter | Typical value | Description |
| :--- | :--- | :--- |
| **Calibration samples** | `~256` | Number of images/steps to analyze |
| **keep_ratio** | `~0.25` | Ratio of layers to keep in FP16 |
| **Optimization steps** | `20–25` | Steps for MSE optimization |

*Exact values may vary depending on the model and dataset.*

---

## Compatibility
* **ComfyUI:** Current mainline
* **Model:** SDXL UNet
* **Environment:** PyTorch with FP8 support (`torch.float8_e4m3fn`)

---

## Relationship to the Original Project
Algorithm credit and design belong entirely to the original author.
This repository exists solely to bridge HSWQ into ComfyUI.
The original implementation remains the authoritative reference.

**Original repository:**
👉 [https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)

### Upstream / Reuse
If any part of this implementation is useful:
* Feel free to reference this repository.
* Parts may be extracted or adapted upstream if desired.
* I am happy to rework parts to better match upstream conventions or extract minimal patches/design notes if helpful.

---

## License
This repository follows the same license terms as the original HSWQ project,
or provides explicit attribution where applicable.
