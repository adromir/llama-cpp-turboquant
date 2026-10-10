# llama.cpp (TurboQuant + AMD ROCm Edition)

<div align="center">

![llama.cpp TurboQuant](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<b>Ultra-compressed KV Cache & Native AMD ROCm Acceleration for Windows and Linux</b>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintainer: Adromir](https://img.shields.io/badge/Maintainer-Adromir-blue.svg)](https://github.com/adromir)
[![ROCm: 10.1.0](https://img.shields.io/badge/ROCm-10.1.0_(TheRock)-red.svg)](https://github.com/adromir/llama-cpp-turboquant)
[![Platform: Windows & Linux](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-brightgreen.svg)](https://github.com/adromir/llama-cpp-turboquant/releases)
[![Architectures: RDNA2 | RDNA3 | RDNA4 | CDNA](https://img.shields.io/badge/GPU%20Targets-RDNA2%20%7C%20RDNA3%20%7C%20RDNA4%20%7C%20CDNA-orange.svg)](https://github.com/adromir/llama-cpp-turboquant)
[![Benchmark Dashboard](https://img.shields.io/badge/Benchmarks-GitHub%20Pages-blueviolet.svg)](https://adromir.github.io/llama-cpp-turboquant/)

[Quick Start](#quick-start) | [Live Benchmark Dashboard](https://adromir.github.io/llama-cpp-turboquant/) | [TurboQuant KV Cache](#what-is-turboquant) | [ROCmFPX Quantization](#what-is-rocmfpx-fpx) | [Strata MoE Cache & Pre-Seeding](#strata-moe-expert-cache--profile-pre-seeding) | [Benchmark Results](#benchmark-results) | [Quantization Tools](#how-to-create-new-quants) | [Branches & Flavors](#branches-and-flavors) | [Pre-built Releases](#pre-built-releases) | [Build from Source](#build-from-source) | [License](#license--credits)

</div>

---

## Overview

This repository is a downstream distribution of [llama.cpp](https://github.com/ggml-org/llama.cpp) maintained by [Adromir](https://github.com/adromir), integrating the revolutionary **TurboQuant** ultra-compressed KV cache technology with **turnkey AMD ROCm/HIP acceleration for Windows and Linux**.

### Why Use This Fork?

1. **Massive KV Cache Memory Savings (TurboQuant)**:
   Compress your KV cache down to **2, 3, or 4 bits per value** (compared to standard FP16 or Q8_0) using orthonormal Walsh-Hadamard Transform (WHT) rotations. Run huge context lengths (32k, 64k, 128k+) on consumer VRAM without severe perplexity degradation.
2. **True Out-of-the-Box Windows & Linux ROCm Execution**:
   Pre-built releases come fully bundled with AMD ROCm 10.1.0 (TheRock) runtime libraries (`rocblas.dll`, `libhipblaslt.dll`, `amdhip64.dll`, etc.). No need to install massive multi-gigabyte AMD ROCm SDKs or configure complex compiler paths.
3. **Universal AMD GPU Architecture Support**:
   Fatbin binaries are pre-compiled for all modern AMD GPU architectures:
   - **RDNA4**: `gfx1200`, `gfx1201` (Radeon RX 9000 series)
   - **RDNA3 / RDNA3.5**: `gfx1100`, `gfx1101`, `gfx1102` (RX 7900, 7800, 7700, 7600, Strix Point)
   - **RDNA2**: `gfx1030` (RX 6900, 6800, 6700)
   - **CDNA / GCN**: `gfx900`, `gfx906`, `gfx908`, `gfx90a` (MI50, MI100, MI200)
4. **Strata MoE Expert Pre-Seeding & Safe Memory Tiering**:
   Ported from [Niko1221/Strata](https://github.com/Niko1221/Strata), Mixture-of-Experts models (such as Qwen 3.8 Next, Mixtral) benefit from dynamic GPU/CPU expert cache tiering. Synchronous binary profile pre-seeding (`--moe-expert-profile`) eliminates first-token latency penalties, delivering ~70%+ cache hits from token 1. Safe host memory pinning (`--moe-cache-pin`) automatically enforces a 4 GiB unpinned OS memory headroom to prevent system thrashing and OOM freezes.
5. **Active Upstream Sync**:
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

## Strata MoE Expert Cache & Profile Pre-Seeding

Mixture-of-Experts (MoE) architectures (such as Qwen 3.8 Next, Mixtral, and DeepSeek) route each token through a small subset of expert layers. Offloading all expert weights onto consumer GPUs often exceeds VRAM capacity.

Ported from **Strata** ([Niko1221/Strata](https://github.com/Niko1221/Strata)), this distribution integrates high-performance dynamic MoE memory tiering with **startup profile pre-seeding** and **safe host memory pinning**:

```
                       ┌──────────────────────────────────────┐
                       │  Binary Profile (STRP Format)        │
                       │  data/expert-profile.bin             │
                       └──────────────────┬───────────────────┘
                                          │ Startup Pre-seed
                                          ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Host RAM (Pinned DMA Pool)             GPU VRAM (Device Memory)        │
  │ Cold & Infrequent Experts              Hot & Pre-seeded Experts        │
  │ ┌──────────────┐                       ┌──────────────┐                │
  │ │ Expert 47    │ ────── PCIe DMA ────> │ Expert 0     │ (Pre-seeded)   │
  │ │ Expert 12    │ <── LRU Eviction ──── │ Expert 3     │ (Pre-seeded)   │
  │ └──────────────┘                       └──────────────┘                │
  │                                                                        │
  │ [Safety Guard]: Minimum 4 GiB unpinned RAM reserved for Windows/Linux  │
  └────────────────────────────────────────────────────────────────────────┘
```

### 1. Instant Cache Warmup via Expert Profiles (`--moe-expert-profile`)

Standard LRU MoE caches start completely empty. During the first hundred tokens, cache misses force continuous synchronous PCIe weight transfers, causing sluggish initial generation and erratic latency.

With `--moe-expert-profile`, the engine reads an activation profile in binary `STRP` format at startup and loads the most frequently activated experts directly into GPU VRAM before inference begins:
- **Instant ~70%+ Cache Hit Rate**: Eliminates cold-start warmup latency from the very first token.
- **Included Profiles**:
  - `data/expert-profile.bin`: Calibrated for general multi-turn instruction following, reasoning, and chat.
  - `data/expert-profile-coder.bin`: Calibrated for programming, syntax comprehension, and technical tasks.
- **Custom Profile Generation**:
  Generate your own profiles using `tools/make_profile.py`:
  ```bash
  python tools/make_profile.py --input activations.csv --output data/custom-profile.bin
  ```

### 2. Safe Host Memory Pinning (`--moe-cache-pin`)

Page-locked (pinned) host memory enables direct DMA transfers across PCIe without intermediate CPU copy overhead, maximizing transfer speed when streaming experts into VRAM. However, excessive memory pinning on 32 GB or 48 GB host systems can cause kernel memory exhaustion, severe OS thrashing, or system lockups.

Our implementation includes **automatic OS memory headroom protection** (via `GlobalMemoryStatusEx` on Windows and `sysinfo` on Linux):
- Dynamically queries total and available physical RAM before pinning.
- Enforces a strict minimum **4 GiB unpinned headroom** reserved for the operating system and background applications.
- If allocating pinned memory would leave less than 4 GiB free, memory pinning is automatically skipped with an informational log: `moe_cache: skipping host pinning to preserve 4 GiB system RAM headroom`.
- Controlled explicitly via `--moe-cache-pin` (default enabled with safety guard) or `--no-moe-cache-pin`.

### 3. MoE Execution Example

```bash
# Run Qwen 3.8 Next with 24GB GPU expert cache, profile pre-seeding, and safe host pinning:
llama-cli.exe -m models/Qwen3.8-Next-MoE-Q4_K_M.gguf \
  -c 32768 -ngl 99 -fa 1 \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --moe-expert-cache 24576M \
  --moe-expert-profile data/expert-profile.bin \
  --moe-cache-pin
```

---

## Benchmark Results

A comprehensive 3-way benchmark evaluation was conducted on AMD RDNA 4 hardware comparing upstream `llama.cpp` against this experimental distribution across **prefill throughput**, **decode speed**, **Multi-Token Prediction (MTP) acceptance rate**, **VRAM utilization**, and **maximum viable context length**.

> [!TIP]
> **Live Interactive Benchmark Dashboard (GitHub Pages)**:
> An interactive dashboard with Chart.js visualization, real-time comparisons, Strata MoE profiling, and per-metric breakdowns is hosted live on GitHub Pages:
> - **Live Dashboard**: [https://adromir.github.io/llama-cpp-turboquant/](https://adromir.github.io/llama-cpp-turboquant/)
> - **Source in Repo**: [`docs/index.html`](docs/index.html) or [`docs/benchmark-results.html`](docs/benchmark-results.html)

### Testbed Environment
- **GPU**: AMD Radeon RX 9060 XT 16GB (RDNA 4, `gfx1200`, 16,304 MiB VRAM)
- **CPU**: AMD Ryzen 9 9950X3D 16-Core Processor (32 Threads)
- **OS / Stack**: Windows 11 Pro / AMD ROCm 10.1.0 (TheRock toolchain)
- **Evaluated Model**: `Qwen3.8-27B` (27.32B parameters, 65 layers, 4-head GQA, 1 MTP head)

### 3-Way Comparative Overview

| Metric / Scenario | Config 1: Upstream (Q4_K_M) | Config 2: Experimental (Q4_K_M) | Config 3: Experimental (ROCmFP4_FAST) | Speedup vs. Upstream |
| :--- | :---: | :---: | :---: | :---: |
| **Model VRAM Footprint** | 15.65 GiB | 15.65 GiB | **13.53 GiB** | **-2.12 GiB savings** |
| **VRAM Headroom (16GB GPU)** | ~650 MiB (Severe OOM risk) | ~650 MiB | **~2,770 MiB (Spacious)** | **4.26x more headroom** |
| **Prefill Throughput (p=512)** | 190.4 - 240.9 t/s | 215.6 - 261.4 t/s | **806.49 t/s** | **3.35x - 4.23x faster** |
| **Prefill Throughput (p=2048)** | 226.5 t/s | 239.8 t/s | **780.58 t/s** | **3.45x faster** |
| **Average Prefill (512 - 4096)** | ~215 t/s | ~235 t/s | **538.53 t/s** | **2.50x faster overall** |
| **Decode Speed (Batch=1)** | 10.25 t/s | 8.98 - 9.12 t/s | **19.65 t/s** | **1.92x faster** |
| **Real-World Generation (MTP)** | 12.80 t/s | 11.45 t/s | **27.40 t/s** | **2.14x faster** |
| **MTP Draft Acceptance Rate** | 27.2% - 38.5% | 27.6% - 39.1% | **28.4% - 41.2%** | **High stability across quants** |
| **Max Context on 16GB VRAM** | 2k - 4k tokens | 2k - 4k tokens | **32k+ (with TurboQuant KV)** | **8x - 16x larger context** |

### Key Takeaways

1. **Prefill Throughput (ROCmFP4 MMQ Matrix Multiplication)**:
   By implementing native matrix-quantization (MMQ) dispatch for `Q4_0_ROCMFP4` and `Q4_0_ROCMFP4_FAST`, prefill throughput skyrocketed from 82 t/s to **806.49 t/s**, outperforming Upstream Q4_K_M by up to **3.4x**. Even on unoptimized `Q4_K_M` weights, the experimental RDNA Flash Attention kernels yield an **+8.5% prefill boost** over upstream.
2. **Decode Speed**:
   Native ROCmFP4 Lloyd-Max 4-bit float representations decode at **19.65 t/s**, nearly doubling the 10.25 t/s decode rate of upstream standard K-quants on consumer 16GB GPUs.
3. **Multi-Token Prediction (MTP) Speculative Decoding**:
   Using `--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.00` leverages Qwen 3.8 internal `nextn_predict_layers` head. Empirical sweeps show `--spec-draft-n-max 6` yields a +2.8% to +7.0% decode speedup over the default of 4 across short and long contexts, while disabling probability gating (`--spec-draft-p-min 0.00`) prevents a 4% to 15% decode speed penalty. All configurations achieve high draft acceptance (~28% to 41%), delivering real-world generation rates of up to **27.40 t/s** on ROCmFP4.
4. **VRAM Headroom and Context Scaling**:
   Standard `Q4_K_M` occupies 15.65 GiB of VRAM, leaving less than 700 MiB free on a 16GB card and crashing with out-of-memory errors beyond 4k tokens. `Q4_0_ROCMFP4_FAST` occupies only 13.53 GiB. Furthermore, at deep contexts (128k+ tokens), standard F16 KV cache grows to ~35 GB, exceeding the weight size of the model and causing severe memory paging on consumer GPUs and APUs. Pairing ROCmFP4 model weights with **TurboQuant 3-bit (`turbo3`) KV cache** compresses the cache from ~35 GB down to ~7-9 GB, enabling comfortable **32k to 128k+ context execution without memory thrashing**.

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
# 1. Launch the interactive WPF GUI (includes built-in Imatrix Generator dialog):
.\scripts\quantize-rocmfpx-gui.ps1
# (or run .\scripts\quantize-rocmfpx.ps1 without arguments)

# 2. Basic CLI FP4 quantization:
.\scripts\quantize-rocmfpx.ps1 -Source "models\model-f16.gguf" -Output "models\model-rocmfp4.gguf" -Preset Q4_0_ROCMFP4_FAST

# 3. All-in-one Imatrix calculation & 3-bit Agent quantization:
.\scripts\quantize-rocmfpx.ps1 -Source "models\model-f16.gguf" -CalibrationData "data\calibration.txt" -Preset Q3_0_ROCMFPX_AGENT

# 4. Quantize using an existing importance matrix:
.\scripts\quantize-rocmfpx.ps1 -Source "models\model-f16.gguf" -Output "models\model-rocmfp3-agent.gguf" -Preset Q3_0_ROCMFPX_AGENT -Imatrix "models\imatrix.gguf"

# 5. Requantizing from an existing Q8_0 GGUF:
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
> For extreme low-bit formats (`Q3_0_ROCMFPX`, `tq3_1s`), passing an existing importance matrix (`--imatrix <path>`) helps preserve syntax tracking and reasoning stability.

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

- `llama-rocm-vanilla-windows.zip` / `llama-rocm-vanilla-linux.zip`: Stable TurboQuant builds (universal AVX2 compatibility).
- `llama-rocm-experimental-windows.zip` / `llama-rocm-experimental-linux.zip`: RDNA boosts, ROCmFPX, and DFlash2 experimental builds (universal AVX2 compatibility for all modern x86-64 CPUs).
- `llama-rocm-experimental-avx512-windows.zip` / `llama-rocm-experimental-avx512-linux.zip`: Specialized **AVX-512** experimental builds (compiled with AVX512F, VBMI, VNNI, BF16). Unlocks maximum performance for **hybrid CPU+GPU offloading** (e.g. Qwen 3.8 Flash-Next 51B, 70B+ models exceeding VRAM) and 2x-3x faster model quantization on AMD Zen 4 / Zen 5 (e.g. Ryzen 9 9950X3D, 7000/9000 series) and modern Intel CPUs.
- `llama-rocm-experimental-scripts-windows.zip`: Standalone PowerShell quantization scripts (`quantize-rocmfpx.ps1`, `quantize-rocmfpx-agent.ps1`, etc.) for Windows.
- `llama-rocm-experimental-scripts-linux.zip`: Standalone Bash quantization scripts (`quantize-rocmfpx-agent.sh`, `quantize-rocmfpx-from-kquant.sh`, etc.) for Linux.

### Installation

1. Download the zip archive for your operating system and CPU architecture from the latest release.
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

### 3. Mixture-of-Experts (MoE) with Strata Profile Pre-Seeding & Safe Pinning

Accelerate MoE inference (e.g. Qwen 3.8 Next, Mixtral) with GPU expert caching, instant startup pre-seeding, and safe host memory pinning:

```bash
# Windows
llama-cli.exe -m models/Qwen3.8-Next-MoE-Q4_K_M.gguf -c 32768 -ngl 99 -fa 1 --cache-type-k turbo3 --cache-type-v turbo3 --moe-expert-cache 24576M --moe-expert-profile data/expert-profile.bin --moe-cache-pin

# Linux
./llama-cli -m models/Qwen3.8-Next-MoE-Q4_K_M.gguf -c 32768 -ngl 99 -fa 1 --cache-type-k turbo3 --cache-type-v turbo3 --moe-expert-cache 24576M --moe-expert-profile data/expert-profile.bin --moe-cache-pin
```

### 4. OpenAI-Compatible API Server (with MTP Speculative Decoding)

Launch the web server with MTP speculative decoding, TurboQuant KV cache, and optimized draft parameters on port 8080:

```bash
llama-server -m models/model.gguf -c 32768 -ngl 99 -fa 1 --cache-type-k q8_0 --cache-type-v turbo3 --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.00 --host 0.0.0.0 --port 8080
```

### 5. Benchmark Performance

Benchmark token processing and generation speeds across cache types:

```bash
llama-bench -m models/model.gguf -ngl 99 -fa 1 -p 512,2048 -n 128 -ctk turbo3 -ctv turbo3
```

---

## Key CLI Flags & Runtime Knobs

### Fork CLI Flags

| CLI Flag | Arguments | Default | Description |
| :--- | :--- | :---: | :--- |
| `--cache-type-k` / `-ctk` | `turbo2`, `turbo3`, `turbo4` | `f16` | TurboQuant KV cache quantization type for Keys |
| `--cache-type-v` / `-ctv` | `turbo2`, `turbo3`, `turbo4` | `f16` | TurboQuant KV cache quantization type for Values |
| `--moe-expert-cache` | Bytes / `24576M` / `8G` | `0` | GPU VRAM budget reserved for dynamic MoE expert caching |
| `--moe-expert-profile` | File path (`.bin`) | None | Pre-seeds MoE GPU cache with top experts from binary STRP profile |
| `--moe-cache-pin` | `--moe-cache-pin` / `--no-moe-cache-pin` | Enabled | Enables safe host memory pinning with 4 GiB OS RAM headroom guard |
| `--spec-type` | `draft-mtp` | None | Multi-Token Prediction (MTP) speculative decoding |
| `--spec-draft-n-max` | Integer (e.g. `6`) | `4` | Maximum speculative draft tokens per step |
| `--spec-draft-p-min` | Float (`0.00`) | `0.75` | Probability gating threshold (`0.00` recommended for MTP) |

### TurboQuant Runtime Knobs

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

### Windows (AMD ROCm 10 / HIP)

#### Option A: Automated Build Script (`build.ps1`)

The repository includes an automated build script supporting both standard AVX2 and specialized AVX-512 architectures:

```powershell
# Interactive build (prompts for local branch and CPU architecture)
.\build.ps1

# Directly build with specialized AVX-512 target (Zen 4/5, modern Intel)
.\build.ps1 -Avx512
```

#### Option B: Manual CMake & Ninja Build

You can build with AMD ROCm 10 (TheRock), official AMD ROCm 6.x+, or any custom installation.
The build snippet automatically detects your ROCm root from `$env:HIP_PATH` or `$env:ROCM_PATH`, or allows defining your custom path:

```powershell
# Auto-detect ROCm installation path, or set custom path
$RocmPath = if ($env:HIP_PATH) { $env:HIP_PATH } elseif ($env:ROCM_PATH) { $env:ROCM_PATH } else { "C:/TheRock/build" }
$ClangBin = "$RocmPath/lib/llvm/bin"

mkdir build
cd build

# Universal build (Standard AVX2):
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

# Or for Specialized AVX-512 (Zen 4/5, modern Intel), append:
#   -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON

cmake --build . --config Release --parallel
```

> [!WARNING]
> **Windows MSVC Toolset Compatibility with ROCm Clang**:
> If building with Visual Studio 2026 / 2022, use MSVC toolset version **14.44** (`-T v144` or installed MSVC 14.44). MSVC 14.51 introduced a `_CLANG_BUILTIN` block in `<cmath>` declaring `isgreater/isless/...` as builtins, causing ROCm clang's CUDA/HIP math headers to fail with `"__device__ function cannot overload __host__ __device__ function"`. MSVC 14.44 compiles cleanly without this conflict.

### Linux (ROCm / HIP)

```bash
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=${HIP_PATH:-/opt/rocm}

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

## AMD ROCm Runtime & Environment Configuration

The ROCm backend supports automatic detection across diverse environments, but offers fine-grained control via standard environment variables:

| Environment Variable | Description | Example / Default |
| :--- | :--- | :--- |
| `HIP_PATH` / `ROCM_PATH` | Base path of the AMD ROCm SDK installation. Used by the build system and runtime path discovery. | `C:\TheRock\build` or `C:\Program Files\AMD\ROCm\6.2` or `/opt/rocm` |
| `HIPBLASLT_TENSILE_PATH` | Path to hipBLASLt Tensile library kernels (`TensileLibrary_*.dat.zlib`). **Auto-discovered** from `$HIP_PATH\bin\hipblaslt\library` or `$HIP_PATH\lib\hipblaslt\library`. Can be manually overridden if using custom kernel packaging. | `$env:HIPBLASLT_TENSILE_PATH = "D:\CustomTensile"` |
| `HIP_VISIBLE_DEVICES` | Controls which AMD GPU devices are visible to the process. Useful on systems with both integrated APUs and discrete GPUs. | `1` (Discrete GPU), `0` (APU/iGPU), `0,1` (All), `-1` (CPU Isolation) |
| `GGML_CUDA_GRAPHS` / `GGML_HIP_GRAPHS` | Enables HIP/CUDA Graph execution during single-token autoregressive decoding to eliminate CPU submission jitter. | `1` (default ON) |
| `GGML_TQ_NATIVE` | Native TQ DP4A matrix-vector decode kernels without runtime dequantization. | `1` |

### High-Performance `hipBLASLt` GEMM & Shape Cache

On RDNA3, RDNA3.5, RDNA4, and CDNA, this distribution automatically links and utilizes `hipBLASLt` (the AMD counterpart to NVIDIA cuBLASLt) for single and batched matrix multiplications:
- **Heuristic Kernel Search**: Dynamically selects the fastest TensileLt tile configurations for the active matrix dimensions with up to 32 MB workspace.
- **In-Memory Algorithm Shape Cache**: Caches selected kernel configurations across step evaluations, completely eliminating heuristic query latency during inference.
- **Strided Batched GEMM**: Native acceleration for multi-head attention projections and batched GEMMs via `HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT` and `HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`.
- **Fail-Safe Fallback**: Any non-standard shapes or unsupported dimension splits automatically and transparently fall back to `cublasGemmEx` / `cublasGemmStridedBatchedEx` / `cublasSgemm` with zero regression.

### AMD Hardware, BIOS & Long-Context Tuning (128k+)

Empirical profiling on AMD RDNA and unified APU architectures (such as Strix Halo / Radeon 8060S and desktop RDNA4) reveals critical settings for long context and speculative execution:

- **IOMMU Disabled in BIOS (+40% Prefill at 128k)**:
  On systems with unified memory architectures, disabling IOMMU in the motherboard BIOS (or booting with `amd_iommu=off` on Linux) reduces memory address translation overhead. This yields up to a **+40% prefill speedup at 128k context** (+1% at 4k, +6% at 16k, +11% to +21% at 32k) with zero degradation to decode speed.
- **MTP Draft Depth (`--spec-draft-n-max 6`)**:
  Empirical sweeps show draft depth `6` provides the best balance between draft acceptance and decode overhead (+2.8% to +7% decode boost over the default of 4). Setting `n-max 8` gains marginally at 32k but regresses on short prompts.
- **Disable Probability Gating (`--spec-draft-p-min 0.00`)**:
  Always keep `p-min` at `0.00` for MTP in `llama.cpp`. Adding draft probability filters (e.g. 0.75) causes a 4% to 15% decode speed loss due to prematurely truncated draft sequences.
- **Unified Memory Split on 128GB APUs**:
  On 128GB unified APUs, a balanced 64 GB host / 64 GB VRAM BIOS partition outperforms 96 GB / 32 GB for deep contexts because Windows ROCm allocations for large KV caches spill into host memory space.

---

## Testing Gates

This repository enforces strict numerical correctness and basis tests before releases:

- `test-turbo-quant`: Turbo basis MSE = 0.0, Cosine = 1.0, and chunked dequantization invariance.
- `test-quantize-fns`: Validates Lloyd-Max round-trip error budgets on `TQ3_1S` and `TQ4_1S`.
- `test-moe-cache`: Validates binary STRP profile parsing, top-K pre-seeding, LRU eviction correctness, and OS memory headroom safety margins.
- `test-backend-ops`: Numerical verification of per-op GGML graphs between CPU and AMD ROCm GPU backend across all operators (`FLASH_ATTN_EXT`, `MUL_MAT`, `SET_ROWS`, `CPY`).

---

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied. Experimental branches and features (`experiment/rdna-boosts`) are under active development and may undergo breaking changes. Please verify models and outputs before deploying in production environments.

---

## License & Credits

This project is licensed under the [MIT License](LICENSE).

**Maintainer**: [Adromir](https://github.com/adromir)  
**Project Repository**: [https://github.com/adromir/llama-cpp-turboquant](https://github.com/adromir/llama-cpp-turboquant)  

### Integrated Community Forks & Upstream Projects

This distribution incorporates features, kernel optimizations, and architectural designs from several outstanding community forks and upstream repositories:

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**: The official upstream source of truth for the ggml tensor library, llama architecture, and model inference engine.
- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)**: Tom Turney's revolutionary TurboQuant KV-cache quantization codec, Walsh-Hadamard Transform (WHT) orthonormal rotation, Lloyd-Max centroid optimization, and elementwise chain fusion.
- **[Niko1221/Strata](https://github.com/Niko1221/Strata)**: Specialized Mixture-of-Experts architecture for Qwen 3.8 Next, dynamic GPU/CPU expert cache tiering, binary STRP profile pre-seeding, and host memory pinning.
- **[stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)** & **[stew675/llama.cpp](https://github.com/stew675/llama.cpp)**: Stew Forster's comprehensive AMD RDNA optimization suite: native-BF16 Flash Attention tiles, RDNA4 WMMA tensor core acceleration, fused chunked Gated-Delta-Net, fused MoE gate+up GLU kernels, and adaptive MTP speculative decoding.
- **[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)**: Carlo Pasquale's high-performance ROCmFPX sub-8-bit floating-point and integer quantization family (`Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`, `Q3_0_ROCMFPX`, `Q6_0_ROCMFPX`, `Q8_0_ROCMFPX`, `Q2_0_ROCMFPX`, `Q4_0_ROCMI4`) for AMD RDNA and CDNA architectures.
- **[daimonionnn/amd-rocmfpx-for-win](https://github.com/daimonionnn/amd-rocmfpx-for-win)**: Empirical profiling, MTP draft tuning benchmarks (`--spec-draft-n-max 6`), and deep-context hardware optimization findings on AMD Strix Halo / ROCm Windows.
- **[unslothai/llama.cpp](https://github.com/unslothai/llama.cpp)**: Unsloth AI optimizations including shape-aware CUDA/HIP graph keying, contiguous virtual memory run mapping, batched VM readahead, and MTP shared tensor borrowing.

