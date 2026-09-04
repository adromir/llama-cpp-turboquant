# llama.cpp (TurboQuant + AMD ROCm Edition)

<div align="center">

![llama.cpp TurboQuant](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<b>Ultra-compressed KV Cache & Native AMD ROCm Acceleration for Windows and Linux</b>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author: Adromir](https://img.shields.io/badge/Author-Adromir-blue.svg)](https://github.com/adromir)
[![ROCm: 10.0.0](https://img.shields.io/badge/ROCm-10.0.0_(TheRock)-red.svg)](https://github.com/adromir/llama-cpp-turboquant)
[![Platform: Windows & Linux](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-brightgreen.svg)](https://github.com/adromir/llama-cpp-turboquant/releases)
[![Architectures: RDNA2 | RDNA3 | RDNA4 | CDNA](https://img.shields.io/badge/GPU%20Targets-RDNA2%20%7C%20RDNA3%20%7C%20RDNA4%20%7C%20CDNA-orange.svg)](https://github.com/adromir/llama-cpp-turboquant)

[Quick Start](#quick-start) | [What is TurboQuant?](#what-is-turboquant) | [What is ROCmFPX?](#what-is-rocmfpx-fpx) | [Create Quants & Imatrix](#how-to-create-new-quants) | [Branches & Flavors](#branches-and-flavors) | [Pre-built Releases](#pre-built-releases) | [Build from Source](#build-from-source) | [License](#license)

</div>

---

## Overview

This repository is a downstream distribution of [llama.cpp](https://github.com/ggml-org/llama.cpp) maintained by [Adromir](https://github.com/adromir), integrating the revolutionary **TurboQuant** ultra-compressed KV cache technology with **turnkey AMD ROCm/HIP acceleration for Windows and Linux**.

### Why Use This Fork?

1. **Massive KV Cache Memory Savings (TurboQuant)**:
   Compress your KV cache down to **2, 3, or 4 bits per value** (compared to standard FP16 or Q8_0) using orthonormal Walsh-Hadamard Transform (WHT) rotations. Run huge context lengths (32k, 64k, 128k+) on consumer VRAM without severe perplexity degradation.
2. **True Out-of-the-Box Windows & Linux ROCm Execution**:
   Pre-built releases come fully bundled with AMD ROCm 10.0.0 (TheRock) runtime libraries (`rocblas.dll`, `libhipblaslt.dll`, `amdhip64.dll`, etc.). No need to install massive multi-gigabyte AMD ROCm SDKs or configure complex compiler paths.
3. **Universal AMD GPU Architecture Support**:
   Fatbin binaries are pre-compiled for all modern AMD GPU architectures:
   - **RDNA4**: `gfx1200`, `gfx1201` (Radeon RX 9000 series)
   - **RDNA3 / RDNA3.5**: `gfx1100`, `gfx1101`, `gfx1102` (RX 7900, 7800, 7700, 7600, Strix Point)
   - **RDNA2**: `gfx1030` (RX 6900, 6800, 6700)
   - **CDNA / GCN**: `gfx900`, `gfx906`, `gfx908`, `gfx90a` (MI50, MI100, MI200)
4. **Active Upstream Sync**:
   Tracks upstream `ggml-org/llama.cpp` and `TheTom/llama-cpp-turboquant` to provide the latest model architectures, sampling improvements, and performance patches.

---

## What is TurboQuant?

TurboQuant is a KV cache quantization codec developed by Tom Turney ([TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)).

Standard quantization methods struggle with outlier activations in the Key and Value vectors. TurboQuant applies a fixed 128x128 orthonormal Walsh-Hadamard rotation (`GGML_OP_TURBO_WHT`) before quantizing to cache memory. This Gaussianizes the activation distribution, eliminates outliers, and applies an inverse-WHT rotation during Flash Attention dequantization.

### TurboQuant Cache & Weight Types

| Type | GGML Type | Purpose | Effective Size | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `turbo2` | `GGML_TYPE_TURBO2_0` | KV cache only | 2.00 bits / value | Extreme compression for 64k-128k+ contexts |
| `turbo3` | `GGML_TYPE_TURBO3_0` | KV cache only | 3.25 bits / value | Balanced precision / compression (recommended) |
| `turbo4` | `GGML_TYPE_TURBO4_0` | KV cache only | 4.25 bits / value | Near-lossless KV cache quality |
| `tq3_1s` | `GGML_TYPE_TQ3_1S` | Model weights | 3.00 bits (block 32)| WHT-rotated Lloyd-Max quantized weights |
| `tq4_1s` | `GGML_TYPE_TQ4_1S` | Model weights | 4.00 bits (block 32)| Native warp-cooperative mmvq weights |

> [!NOTE]
> Turbo cache types require Flash Attention (`-fa 1`), which is automatically enabled. DeepSeek/MLA models do not have a separate V cache, so identical K and V types should be used.

---

## What is ROCmFPX (FPX)?

**ROCmFPX** (developed by Carlo Pasquale / [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)) is a high-performance sub-8-bit floating-point and integer quantization family engineered specifically for AMD GPU hardware (RDNA2, RDNA3, RDNA3.5, RDNA4, and Strix Halo APUs).

Unlike conventional integer k-quants (`Q4_K_M`, `Q5_K_M`), ROCmFPX formats use native floating-point and integer encodings that map directly to AMD SIMD and WMMA matrix units for blistering decode speeds and compact model footprints:

### ROCmFPX Quantization Formats

| Format | GGML Type | Precision | Effective BPW | Target Use Case & Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| `Q4_0_ROCMFP4` | `GGML_TYPE_Q4_0_ROCMFP4` | 4-bit Float (E2M1) | ~4.50 | Standard 4-bit float format with balanced perplexity |
| `Q4_0_ROCMFP4_FAST` | `GGML_TYPE_Q4_0_ROCMFP4_FAST` | 4-bit Float (E2M1) | ~4.50 | Maximum decode tok/s on AMD RDNA GPUs (recommended FP4) |
| `Q3_0_ROCMFPX` | `GGML_TYPE_Q3_0_ROCMFPX` | 3-bit Float (FP3) | ~3.30 | Ultra-compact 3-bit weights for large models on smaller VRAM |
| `Q6_0_ROCMFPX` | `GGML_TYPE_Q6_0_ROCMFPX` | 6-bit Float (E3M2) | ~6.50 | Near-F16 accuracy with 25% memory savings compared to Q8_0 |
| `Q8_0_ROCMFPX` | `GGML_TYPE_Q8_0_ROCMFPX` | 8-bit Float (FP8) | ~8.50 | Reference-grade precision for base models and critical layers |
| `Q4_0_ROCMI4` | `GGML_TYPE_Q4_0_ROCMI4` | 4-bit Int (W4A4) | ~4.00 | Experimental W4A4 integer MMQ acceleration on RDNA3.5/RDNA4 |

### Agent & Coherent Presets

For production agents requiring strict JSON formatting, tool calling, or complex reasoning, standard aggressive quantization can cause occasional syntax errors. ROCmFPX provides **Agent / Coherent presets**:
- `Q4_0_ROCMFP4_COHERENT`: Keeps output layers, embeddings, and sensitive attention heads at `Q6_K` / `Q8_0` while quantizing dense MLP weights to ROCmFP4.
- `Q3_0_ROCMFPX_AGENT`: Coherent 3-bit quantization preserving JSON syntax tracking.
- `Q6_0_ROCMFPX_AGENT`: Near-lossless agent execution with high context stability.

---

## How to Create New Quants

You can quantize any model from standard BF16/F16 GGUF weights or requantize from existing `Q4_K_M`/`Q8_0` files.

> [!NOTE]
> The automated quantization helper scripts are maintained on the [`experiment/rdna-boosts`](https://github.com/adromir/llama-cpp-turboquant/tree/experiment/rdna-boosts) branch and are also distributed as standalone zip archives (`llama-rocm-experimental-scripts-windows.zip` and `llama-rocm-experimental-scripts-linux.zip`) in our [Releases](https://github.com/adromir/llama-cpp-turboquant/releases).

### Method 1: Direct CLI (`llama-quantize`)

`llama-quantize` natively supports both TurboQuant weight types (`tq3_1s`, `tq4_1s`) and ROCmFPX formats:

```bash
# 1. Quantize from BF16/F16 to ROCmFP4 (Fast)
llama-quantize models/model-BF16.gguf models/model-ROCmFP4.gguf Q4_0_ROCMFP4_FAST

# 2. Quantize to 3-bit ROCmFP3
llama-quantize models/model-BF16.gguf models/model-ROCmFP3.gguf Q3_0_ROCMFPX

# 3. Quantize to TurboQuant WHT-Rotated Weights
llama-quantize models/model-BF16.gguf models/model-TQ4.gguf tq4_1s

# 4. Requantize from an existing Q8_0 or Q4_K_M GGUF (add --allow-requantize)
llama-quantize --allow-requantize models/model-Q8_0.gguf models/model-ROCmFP4.gguf Q4_0_ROCMFP4_FAST

# 5. Using an Importance Matrix (Imatrix) for superior quality at low bitrates
llama-quantize --imatrix imatrix.gguf models/model-BF16.gguf models/model-ROCmFP3-imatrix.gguf Q3_0_ROCMFPX
```

### Method 2: Windows PowerShell & WPF GUI (`quantize-rocmfpx-gui.ps1`)

For Windows users, we provide both an interactive WPF graphical interface and a CLI helper script:

```powershell
# 1. Launch the interactive WPF GUI (auto-detects llama-quantize.exe):
.\scripts\quantize-rocmfpx-gui.ps1
# (or double-click / run .\scripts\quantize-rocmfpx.ps1 without arguments)

# 2. Basic CLI FP4 quantization:
.\scripts\quantize-rocmfpx.ps1 -Source "models\model-f16.gguf" -Output "models\model-rocmfp4.gguf" -Preset Q4_0_ROCMFP4_FAST

# 3-bit Agent quantization with importance matrix
.\scripts\quantize-rocmfpx.ps1 -Source "models\model-f16.gguf" -Output "models\model-rocmfp3-agent.gguf" -Preset Q3_0_ROCMFPX_AGENT -Imatrix "models\imatrix.gguf"

# Requantizing from an existing Q8_0 GGUF
.\scripts\quantize-rocmfpx.ps1 -Source "models\model-Q8_0.gguf" -Output "models\model-rocmfp6.gguf" -Preset Q6_0_ROCMFPX -AllowRequantize
```

### Method 3: Linux / macOS Bash Scripts

On Linux or macOS, use the dedicated bash scripts in `scripts/`:

```bash
# Quantize BF16 to ROCmFP4 using the agent profile
SRC=model-BF16.gguf OUT=model-ROCmFP4-agent.gguf FORMAT=rocmfp4 PROFILE=agent ./scripts/quantize-rocmfpx-agent.sh

# Quantize to fast ROCmFP4
SRC=model-BF16.gguf OUT=model-ROCmFP4-fast.gguf FORMAT=rocmfp4 PROFILE=fast ./scripts/quantize-rocmfpx-agent.sh

# Requantize from an existing K-quant (e.g. Q4_K_M or Q8_0)
SRC=model-Q4_K_M.gguf OUT=model-Q3_0_ROCMFPX.gguf PRESET=Q3_0_ROCMFPX ./scripts/quantize-rocmfpx-from-kquant.sh
```

> [!TIP]
> **Quality Ladder for Requantizing**:
> When original BF16 sources are not available, use the highest quality source possible:
> `BF16/F16` (Best) > `Q8_0` > `Q6_K` > `Q4_K_M` (Acceptable floor for Q3).
> Never requantize an existing ROCmFPX file into another ROCmFPX format (double-quantization causes severe degradation).

---

### Understanding the Importance Matrix (imatrix)

An **Importance Matrix (`imatrix`)** is a powerful calibration technique in `llama.cpp` that dramatically improves quantization quality, especially for low-bit formats (`Q3_0_ROCMFPX`, `tq3_1s`, `Q4_0_ROCMFP4`, `IQ3_XXS`, `Q4_K_M`).

#### How it Works: Uniform MSE vs. Weighted MSE
- **Standard Quantization (Uniform MSE)**: Minimizes rounding error equally across all tensor coordinates:
  $$\min \sum (W_{ij} - \hat{W}_{ij})^2$$
  This treats inactive weights and critical attention channels with the exact same priority.
- **imatrix Quantization (Weighted MSE)**: Feeds a calibration text dataset through the unquantized model to compute the actual activation variance ($I_{ij} \approx \sum A_{ik}^2$) flowing through every channel:
  $$\min \sum I_{ij} \cdot (W_{ij} - \hat{W}_{ij})^2$$
  Weights that experience massive activation spikes or carry high semantic influence receive maximum quantization fidelity, while less critical weights absorb the quantization noise.

#### Why Use an Imatrix?
| Quantization Level | Without Imatrix | With Imatrix | Real-world Impact |
| :--- | :--- | :--- | :--- |
| **8-Bit (`Q8_0`, `FP8`)** | Excellent | Near-Lossless | Negligible difference (quantization noise is already minimal) |
| **6-Bit (`Q6_K`, `FP6`)** | Very Good | Near-Lossless | Slight perplexity gain (~0.02 PPL) |
| **4-Bit (`ROCmFP4`, `Q4_K_M`)** | Good | Excellent | Measurable uplift (~0.1 - 0.2 PPL), reaches near-FP16 quality |
| **3-Bit (`ROCmFP3`, `tq3_1s`)** | Risk of degradation | Coherent & Usable | **Crucial:** Prevents syntax errors, hallucination loops, and severe logic decay |

#### Step-by-Step: How to Generate and Use an Imatrix

1. **Prepare Calibration Data**:
   Download or create a clean, diverse text file (`calibration.txt`) containing prose, code, math, and JSON (e.g. `groups_merged.txt` or `wiki.train.raw`). A file size of 500 KB to 2 MB is ideal.

2. **Compute the Importance Matrix with `llama-imatrix`**:
   Run `llama-imatrix` with GPU offloading enabled (`-ngl 99`). On modern AMD Radeon GPUs, calibration finishes in just 2 to 5 minutes:
   ```bash
   # Windows
   llama-imatrix.exe -m models/model-BF16.gguf -f data/calibration.txt -o models/imatrix.gguf -ngl 99 -c 2048 --chunks 64

   # Linux
   ./llama-imatrix -m models/model-BF16.gguf -f data/calibration.txt -o models/imatrix.gguf -ngl 99 -c 2048 --chunks 64
   ```
   - `-m`: Path to the unquantized (BF16/F16) model.
   - `-f`: Path to the calibration text dataset.
   - `-o`: Output importance matrix file (`imatrix.gguf`).
   - `-ngl 99`: Offloads layers to AMD ROCm GPU for rapid execution.
   - `-c 2048`: Context window size for computing activation tensors.
   - `--chunks 64`: Number of text chunks to evaluate (64 to 100 chunks is recommended).

3. **Apply the Imatrix during Quantization**:
   Pass `--imatrix` directly into `llama-quantize`, `quantize-rocmfpx.ps1`, or the bash scripts:
   ```bash
   # CLI
   llama-quantize --imatrix models/imatrix.gguf models/model-BF16.gguf models/model-Q3_0_ROCMFPX.gguf Q3_0_ROCMFPX

   # PowerShell (Windows)
   .\scripts\quantize-rocmfpx.ps1 -Source "models\model-BF16.gguf" -Output "models\model-Q3.gguf" -Preset Q3_0_ROCMFPX -Imatrix "models\imatrix.gguf"

   # Bash (Linux)
   IMATRIX=models/imatrix.gguf SRC=models/model-BF16.gguf OUT=models/model-Q3.gguf FORMAT=rocmfp3 PROFILE=agent ./scripts/quantize-rocmfpx-agent.sh
   ```

---

## Branches and Flavors

This repository maintains two distinct build targets published as separate releases:

```
                  ┌─────────────────────────────────────┐
                  │   adromir/llama-cpp-turboquant      │
                  └──────────────────┬──────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │     custom-workflow     │             │ experiment/rdna-boosts  │
    │     (Default Branch)    │             │  (Experimental Branch)  │
    ├─────────────────────────┤             ├─────────────────────────┤
    │ - Stable TurboQuant     │             │ - TurboQuant KV Cache   │
    │ - Upstream synced base  │             │ - Stew's RDNA Boosts    │
    │ - Universal ROCm 10 CI  │             │ - Native-BF16 FlashAttn │
    │                         │             │ - ROCmFPX (FP4/FP8/IU4) │
    │                         │             │ - DFlash2 Speculative   │
    └────────────┬────────────┘             └────────────┬────────────┘
                 ▼                                       ▼
          [tag]-vanilla                           [tag]-experimental
```

1. **`custom-workflow` (Default)**:
   - Contains the vanilla, stable TurboQuant feature set synced with upstream `feature/turboquant-kv-cache`.
   - Automated multi-platform CI packaging for Windows and Linux.
   - Recommended for general production use and maximum compatibility.
2. **`experiment/rdna-boosts`**:
   - Incorporates cutting-edge optimizations for AMD GPUs:
     - **Stew's RDNA Boosts**: Native-BF16 Flash Attention tiles, WMMA tensor core acceleration, fused GDN (Gated Delta Net), fused MoE small-batch paths.
     - **ROCmFPX Family**: Experimental sub-8-bit floating point matrix multiplication routines.
     - **DFlash2 Speculative Decoding**: Draft-free speculative decoding architecture and lattice verification.
     - **Shape-aware Graph Hashing**: Eliminates CUDA/HIP graph warmup resets on variable batch verification.

---

## Pre-built Releases

Pre-compiled, self-contained zip packages for both **Windows** and **Linux** are available under [Releases](https://github.com/adromir/llama-cpp-turboquant/releases):

- `llama-rocm-vanilla-windows.zip` / `llama-rocm-vanilla-linux.zip`: Stable TurboQuant builds.
- `llama-rocm-experimental-windows.zip` / `llama-rocm-experimental-linux.zip`: RDNA boosts, ROCmFPX, and experimental feature builds.
- `llama-rocm-experimental-scripts-windows.zip`: Standalone PowerShell quantization scripts (`quantize-rocmfpx.ps1`, `quantize-rocmfpx-agent.ps1`, etc.) for Windows.
- `llama-rocm-experimental-scripts-linux.zip`: Standalone Bash quantization scripts (`quantize-rocmfpx-agent.sh`, `quantize-rocmfpx-from-kquant.sh`, etc.) for Linux.

### Installation

1. Download the zip archive for your operating system from the latest release.
2. Extract the archive to any folder.
3. Open a terminal in the extracted folder.
4. Run `llama-cli.exe` or `llama-server.exe` directly!

> [!TIP]
> If your system has both an integrated AMD GPU (e.g. `gfx1036`) and a discrete AMD GPU (e.g. `gfx1200` RX 9060 XT), set the visible device in your shell:
> - **PowerShell (Windows)**: `$env:HIP_VISIBLE_DEVICES="1"`
> - **Bash (Linux)**: `export HIP_VISIBLE_DEVICES=1`

---

## Quick Start

### 1. Interactive Chat with TurboQuant KV Cache

Run inference using 3-bit TurboQuant KV cache:

```bash
# Windows
llama-cli.exe -m models/Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 32768 -ngl 99 -fa 1 --cache-type-k turbo3 --cache-type-v turbo3

# Linux
./llama-cli -m models/Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 32768 -ngl 99 -fa 1 --cache-type-k turbo3 --cache-type-v turbo3
```

### 2. Asymmetric KV Cache for Large GQA Models

For models with high Grouped-Query Attention ratios, using `q8_0` for Keys and `turbo3` for Values offers the optimal balance:

```bash
llama-cli -m models/model.gguf -c 65536 -ngl 99 -fa 1 --cache-type-k q8_0 --cache-type-v turbo3
```

### 3. OpenAI-Compatible API Server

Launch the web server with web UI on port 8080:

```bash
llama-server -m models/model.gguf -c 32768 -ngl 99 -fa 1 --cache-type-k turbo3 --cache-type-v turbo3 --host 0.0.0.0 --port 8080
```

### 4. Benchmark Performance

Benchmark token processing and generation speeds across cache types:

```bash
llama-bench -m models/model.gguf -ngl 99 -fa 1 -p 512,2048 -n 128 -ctk turbo3 -ctv turbo3
```

---

## Runtime Environment Knobs

TurboQuant exposes fine-tuning knobs via environment variables:

| Variable | Default | Description |
| :--- | :---: | :--- |
| `TURBO_LAYER_ADAPTIVE` | `0` | Layer-adaptive KV precision; `7` = Boundary V (edge layers Q8_0, inner layers Turbo) |
| `TURBO_AUTO_ASYMMETRIC` | `1` | Automatically configures asymmetric K/V types for large-GQA models |
| `TURBO_SPARSE_V` | `1` | Skips sparse-V dequantization during Flash Attention |
| `GGML_TQ_NATIVE` | `0` | When set to `1`, uses fused native TQ kernels instead of load-time conversion |

---

## Build from Source

If you prefer building from source, ensure you have CMake and Ninja installed.

### Windows (AMD ROCm 10 TheRock)

```powershell
# Set path to AMD ROCm installation
$RocmPath = "C:/TheRock/build"
$ClangBin = "$RocmPath/lib/llvm/bin"

mkdir build
cd build

cmake -G "Ninja" `
  -DCMAKE_C_COMPILER="$ClangBin/clang.exe" `
  -DCMAKE_CXX_COMPILER="$ClangBin/clang++.exe" `
  -DCMAKE_ASM_COMPILER="$ClangBin/clang.exe" `
  -DCMAKE_HIP_COMPILER="$ClangBin/clang++.exe" `
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$RocmPath" `
  -DGGML_HIP=ON `
  -DGPU_TARGETS="gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_SHARED_LIBS=ON `
  -DGGML_STATIC=OFF `
  -DGGML_OPENMP=OFF `
  -DCMAKE_PREFIX_PATH="$RocmPath;$RocmPath/lib/cmake" `
  ..

cmake --build . --config Release --parallel
```

### Linux (ROCm / HIP)

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm

mkdir build && cd build

cmake -G "Ninja" \
  -DCMAKE_C_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang++ \
  -DCMAKE_ASM_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang \
  -DCMAKE_HIP_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=${ROCM_PATH} \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_STATIC=OFF \
  -DGGML_OPENMP=OFF \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH};${ROCM_PATH}/lib/cmake" \
  ..

cmake --build . --config Release --parallel $(nproc)
```

---

## Testing Gates

This repository enforces strict numerical correctness and basis tests before releases:

- `test-turbo-quant`: Turbo basis MSE = 0.0, Cosine = 1.0, and chunked dequantization invariance.
- `test-quantize-fns`: Validates Lloyd-Max round-trip error budgets on `TQ3_1S` and `TQ4_1S`.
- `test-backend-ops`: Numerical verification of per-op GGML graphs between CPU and AMD ROCm GPU backend across all operators (`FLASH_ATTN_EXT`, `MUL_MAT`, `SET_ROWS`, `CPY`).

---

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied. Experimental branches and features (`experiment/rdna-boosts`) are under active development and may undergo breaking changes. Please verify models and outputs before deploying in production environments.

---

## License

This project is licensed under the [MIT License](LICENSE).

**Author**: [Adromir](https://github.com/adromir)  
**Project Repository**: [https://github.com/adromir/llama-cpp-turboquant](https://github.com/adromir/llama-cpp-turboquant)  
**Upstream Projects**:
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
