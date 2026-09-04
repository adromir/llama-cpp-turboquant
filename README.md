# llama.cpp (TurboQuant + RDNA Boosts + ROCmFPX + DFlash2)

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>High-Performance LLM Inference on AMD Radeon & ROCm with Ultra-Compact KV Cache, Native RDNA Boosts, and DFlash2 Speculative Decoding</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/Author-Adromir-orange.svg)](https://github.com/adromir)
[![Branch](https://img.shields.io/badge/branch-experiment%2Frdna--boosts-brightgreen.svg)](https://github.com/adromir/llama-cpp-turboquant/tree/experiment/rdna-boosts)
[![ROCm](https://img.shields.io/badge/ROCm-10.0.0%20(TheRock)-red.svg)](https://repo.radeon.com/rocm/)
[![AMD GPU](https://img.shields.io/badge/AMD%20GPU-RDNA2%20%7C%20RDNA3%20%7C%20RDNA4-purple.svg)](https://github.com/adromir/llama-cpp-turboquant)
[![Releases](https://img.shields.io/github/v/release/adromir/llama-cpp-turboquant)](https://github.com/adromir/llama-cpp-turboquant/releases)

[Features](#key-features) / [Quantization](#quantization-matrix) / [Quick Start](#quick-start) / [Building](#building-from-source) / [Usage](#usage-examples) / [Author](#author--credits)

</div>

---

## Overview

This repository is an experimental, performance-optimized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) combining bleeding-edge LLM acceleration technologies into a single unified tree:

1. **TurboQuant KV Cache**: 2-bit to 4.25-bit KV cache compression via Walsh-Hadamard Transforms (WHT) and PolarQuant, reducing context memory footprint by 2x to 4x compared to standard Q8_0 while retaining near-FP16 attention quality.
2. **RDNA Boosts**: Deep GPU architecture optimizations by Stew Forster (`stew675/llama-cpp-rdna-boosts`), featuring native-BF16 flash attention, RDNA4 WMMA acceleration, fused chunked Gated-Delta-Net, fused MoE gate+up GLU MMQ/MMVQ kernels, and hybrid HIP all-reduce.
3. **ROCmFPX Quantization Family**: Experimental AMD FP4 and FPx quantization formats ported from [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX), featuring UE4M3-scaled low-bit representations (`Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`, `Q3_0_ROCMFPX`, `Q6_0_ROCMFPX`, `Q8_0_ROCMFPX`, `Q2_0_ROCMFPX`, `Q4_0_ROCMI4`) and specialized agent/tool-calling coherence routing recipes.
4. **DFlash2 Speculative Decoding**: High-throughput block-parallel speculative decoding (PR #27816 & PR #28000) featuring dynamic 1D grouped causal convolution, candidate selector transition scoring lattice, M-RoPE multimodal position batching, and CPU confidence pruning.
5. **Shape-Aware CUDA/HIP Graphs & Unified Memory**: O(1) shape-aware graph hashing preventing warmup thrashing during variable speculative batch evaluation, robust unified memory environment validation, and async 2D D2D memory transfers.
6. **AMD ROCm Core SDK 10.0.0 ("TheRock") Support**: Native support for modern ROCm 10 toolchains on both Windows 11 and Ubuntu 24.04, with fast compilation, automated dependency staging, and multi-architecture fat binaries.

---

## Key Features

### 1. Ultra-Compact TurboQuant KV Cache
- **Massive Context Savings**: Run 64k-128k context windows on consumer GPUs without out-of-memory errors.
- **WHT Rotation**: Fixed 128x128 orthonormal Walsh-Hadamard rotation (`GGML_OP_TURBO_WHT`) Gaussianizes activation vectors prior to quantization, minimizing outlier distortion.
- **PolarQuant Codecs**:
  - `turbo2` (43): 2.00 bits/value (extreme context compression).
  - `turbo3` (44): 3.25 bits/value (recommended sweet spot for long context).
  - `turbo4` (47): 4.25 bits/value (near-FP16 cosine similarity > 0.995).
  - `tq3_1s` (45) & `tq4_1s` (46): 3-bit and 4-bit Lloyd-Max weight quantization with fused native decode.

### 2. RDNA Boosts & Hardware Accelerations
- **RDNA4 WMMA & Wave32 Tuning**: Native wave32 flash attention and tile configurations for RDNA4 (Radeon RX 9060 XT, RX 9070 series; `gfx1200`, `gfx1201`).
- **Native-BF16 Flash Attention**: Hardware-native BF16 execution on RDNA3/4 and CDNA accelerators.
- **Fused Chunked Gated-Delta-Net**: End-to-end fused prefill kernel for recurrent and state-space layers.
- **Fused MoE Gate+Up GLU**: CUDA-graph capturable MoE token routing and fused SwiGLU/clamp dispatch (`GGML_GLU_OP_SWIGLU_CLAMP`).
- **Hybrid HIP All-Reduce**: Optimized multi-GPU and split-device tensor-parallel communication.

### 3. ROCmFPX Experimental Quantization Family
- **AMD FP4 Layouts**:
  - `Q4_0_ROCMFP4`: 4.50 bpw UE4M3 dual scales + packed AMD FP4 blocks.
  - `Q4_0_ROCMFP4_FAST`: 4.25 bpw single scale layout for maximum prefill and decode bandwidth.
- **ROCmFPx Reference Layouts**:
  - `Q3_0_ROCMFPX`: 3.50 bpw UE4M3 3-bit representation.
  - `Q6_0_ROCMFPX`: 6.50 bpw UE4M3 6-bit representation.
  - `Q8_0_ROCMFPX`: 8.25 bpw UE4M3 8-bit representation.
  - `Q2_0_ROCMFPX`: 2.50 bpw S40 codebook + dual UE4M3 scales.
  - `Q4_0_ROCMI4`: 4.25 bpw native signed 4-bit integer path (no codebook).
- **Agent and Tool-Calling Routing Recipes**:
  - `Q4_0_ROCMFP4_LEAN`, `Q4_0_ROCMFP4_COHERENT`, `Q4_0_ROCMFP4_STRIX`.
  - `Q3_0_ROCMFPX_AGENT`, `Q6_0_ROCMFPX_AGENT`, `Q6_0_ROCMFPX_LEAN`.

### 4. DFlash2 Speculative Decoding
- **Block-Parallel Drafting**: Multi-token draft generation using dynamic grouped 1D causal convolutions (`build_dflash2_conv`).
- **Candidate Transition Lattice**: Evaluates candidate selector scores `⟨A[p] ⊙ project(h), B[c]⟩ + unary[c]` (`build_dflash2_selector`) for tree speculation.
- **M-RoPE Multimodal Compatibility**: Native support for 4-row position batches for vision-language models (e.g. Qwen2-VL).
- **Lattice Confidence Pruning**: CPU-side lattice walk and `p_min` confidence thresholding without expensive full logit sampling.

### 5. Shape-Aware CUDA/HIP Graphs & Unified Memory
- **O(1) Shape-Aware Graph Caching**: Mixes split root, node count, and dimension strides into a 64-bit key, preventing warmup resets and thrashing during variable batch verify passes.
- **Safe Managed Memory Validation**: Environment variable helper `ggml_cuda_env_enabled` avoids accidental allocation when `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0` or `false`.
- **Async 2D D2D Copies**: Employs `cudaMemcpyDefault` across CUDA, HIP, and MUSA for direct asynchronous 2D transfers without forced synchronization.

---

## Quantization Matrix

### KV Cache Types

| Type | Bits/Val | Flash Attention | Relative Cache Size (vs FP16) | Recommended Use Case |
|---|---|---|---|---|
| `f16` | 16.0 | Standard / Flash | 1.00x | Maximum precision reference |
| `q8_0` | 8.5 | Standard / Flash | 0.53x | Standard quant baseline |
| `turbo4` | 4.25 | Auto-enabled | 0.27x | High-precision long context |
| `turbo3` | 3.25 | Auto-enabled | 0.20x | Best overall memory/quality ratio |
| `turbo2` | 2.00 | Auto-enabled | 0.13x | Extreme context lengths |

### Model Weight Types

| Type | Enum | Size (bpw) | Description |
|---|---|---|---|
| `Q4_0_ROCMFP4` | 100 | 4.50 bpw | AMD FP4 UE4M3 dual-scale experimental layout |
| `Q4_0_ROCMFP4_FAST` | 101 | 4.25 bpw | AMD FP4 single-scale speed layout |
| `Q6_0_ROCMFPX` | 102 | 6.50 bpw | 6-bit UE4M3-scaled format |
| `Q8_0_ROCMFPX` | 103 | 8.25 bpw | 8-bit UE4M3-scaled format |
| `Q3_0_ROCMFPX` | 104 | 3.50 bpw | 3-bit UE4M3-scaled format |
| `Q2_0_ROCMFPX` | 107 | 2.50 bpw | 2-bit S40 codebook with dual UE4M3 scales |
| `Q4_0_ROCMI4` | 108 | 4.25 bpw | Native signed-nibble 4-bit integer weights |
| `TQ3_1S` | 45 | 4.00 bpw | WHT-rotated 8-level Lloyd-Max weights (block 32) |
| `TQ4_1S` | 46 | 5.00 bpw | WHT-rotated 16-level Lloyd-Max weights (block 32) |

---

## Supported Hardware

The build produces a portable multi-architecture binary supporting modern AMD GPUs:

| Generation | GPU Targets | Common Devices |
|---|---|---|
| **RDNA4** | `gfx1200`, `gfx1201` | Radeon RX 9060 XT, RX 9070, RX 9070 XT |
| **RDNA3.5** | `gfx1150`, `gfx1151` | Ryzen AI 300 Series (Strix Point, Strix Halo) |
| **RDNA3** | `gfx1100`, `gfx1101`, `gfx1102` | Radeon RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT, RX 7600 |
| **RDNA2** | `gfx1030` | Radeon RX 6950 XT, RX 6900 XT, RX 6800 XT, RX 6800 |
| **CDNA / CDNA2** | `gfx900`, `gfx906`, `gfx908`, `gfx90a` | Instinct MI25, MI50, MI100, MI210, MI250X, Radeon VII |

---

## Quick Start

### 1. Download Pre-built Releases
Pre-built packages are published via GitHub Actions on the [Releases](https://github.com/adromir/llama-cpp-turboquant/releases) page:
- `llama-rocm-experiment-rdna-boosts-windows.zip` (Windows 11 x64)
- `llama-rocm-experiment-rdna-boosts-linux.zip` (Ubuntu 24.04 x64)

Both packages include required ROCm runtime libraries (`rocblas.dll`/`.so`, `hipblas.dll`/`.so`, `libhipblaslt.dll`/`.so`, `amdhip64.dll`/`.so`) and Tensile kernel caches.

### 2. Run Inference with TurboQuant KV Cache

```bash
# High-speed inference using TurboQuant 3-bit V-cache and Q8_0 K-cache:
llama-cli -m model.gguf -ngl 99 -c 32768 --cache-type-k q8_0 --cache-type-v turbo3 -p "Hello, world!"

# Maximum compression: 2-bit K and V cache:
llama-cli -m model.gguf -ngl 99 -c 65536 --cache-type-k turbo2 --cache-type-v turbo2 -p "Summarize this document: ..."
```

### 3. Quantize Models to ROCmFPX

#### Direct CLI
```bash
# Standard ROCmFP4 quantization:
llama-quantize model-f32.gguf model-q4_rocmfp4.gguf Q4_0_ROCMFP4

# Speed-optimized single-scale ROCmFP4:
llama-quantize model-f32.gguf model-q4_fast.gguf Q4_0_ROCMFP4_FAST

# Agent-optimized recipe (coherent tool-calling with boosted attention layers):
llama-quantize model-f32.gguf model-q4_lean.gguf Q4_0_ROCMFP4_LEAN
```

#### Automated Helper Scripts (Windows & Linux)
We provide automated helper scripts to quantize with pre-tuned agent/speed profiles and requantize existing K-quants:

- **Windows (PowerShell)**:
  ```powershell
  # Quantize BF16 to ROCmFP4 with agent profile:
  .\scripts\quantize-rocmfpx-agent.ps1 -Src model-f16.gguf -Out model-fp4-agent.gguf -Format rocmfp4 -Profile agent

  # Fast decode layout on Strix / RDNA:
  .\scripts\quantize-rocmfpx-agent.ps1 -Src model-f16.gguf -Out model-fp4-fast.gguf -Format rocmfp4 -Profile fast

  # Requantize an existing Q4_K_M or Q8_0 model without original BF16:
  .\scripts\quantize-rocmfpx-from-kquant.ps1 -Src model-Q4_K_M.gguf -Out model-Q3_0_ROCMFPX.gguf
  ```

- **Linux (Bash)**:
  ```bash
  # Quantize BF16 to ROCmFP4 agent preset:
  FORMAT=rocmfp4 PROFILE=agent SRC=model-f16.gguf OUT=model-fp4-agent.gguf ./scripts/quantize-rocmfpx-agent.sh

  # Requantize an existing K-quant:
  SRC=model-Q4_K_M.gguf OUT=model-Q3_0_ROCMFPX.gguf PRESET=Q3_0_ROCMFPX ./scripts/quantize-rocmfpx-from-kquant.sh
  ```

---

## Building from Source

### Windows 11 (Automated with `build.ps1`)

1. Install **AMD ROCm Core SDK 10.0.0 ("TheRock")** to `C:\TheRock\build` (or set `$env:HIP_PATH`).
2. Install dependencies via Chocolatey:
   ```powershell
   choco install cmake ninja openssl ccache -y
   ```
3. Run the automated local build script:
   ```powershell
   .\build.ps1
   ```
   The script configures CMake, compiles with Clang via Ninja, and stages all necessary ROCm runtime DLLs into `build\bin\`.

### Linux (Ubuntu 24.04)

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm

mkdir build && cd build
cmake -G Ninja \
  -DCMAKE_C_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=${ROCM_PATH} \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  ..

ninja -j$(nproc)
```

---

## Usage & Examples

### Starting the OpenAI-Compatible API Server

```bash
llama-server -m model.gguf -ngl 99 -c 32768 \
  --cache-type-k q8_0 --cache-type-v turbo3 \
  --host 0.0.0.0 --port 8080
```

### Benchmarking Throughput

```bash
# Benchmark decode and prompt-processing across KV cache formats:
llama-bench -m model.gguf -p 512,2048 -n 128 -ctk q8_0 -ctv turbo3,turbo4,f16
```

### Multi-GPU Selection

```powershell
# Target discrete AMD Radeon GPU when an integrated GPU is present:
$env:HIP_VISIBLE_DEVICES = "1"
.\build\bin\llama-cli.exe -m model.gguf -ngl 99
```

### Speculative Decoding (DFlash2)

```bash
# High-speed speculative decoding with DFlash2 draft model and TurboQuant KV cache:
llama-cli -m target-model.gguf -md dflash-draft-model.gguf -ngl 99 -c 8192 \
  --cache-type-k q8_0 --cache-type-v turbo3 \
  -p "Implement a fast Walsh-Hadamard transform in C++:"
```

---

## Screenshots

<table align="center">
    <tr>
        <td align="center" width="50%">
            <img width="1310" height="888" alt="VLM session with llama cli" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <br />
            <i>Interactive multimodal session with <b>llama-cli</b></i>
        </td>
        <td align="center" width="50%">
            <img width="1392" height="958" alt="Built-in web UI against llama serve" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <br />
            <i>Built-in Web UI connected to <b>llama-server</b></i>
        </td>
    </tr>
</table>

---

## Author & Credits

- **Author**: [Adromir](https://github.com/adromir)
- **Website**: [https://github.com/adromir](https://github.com/adromir)
- **Upstream Project**: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.
- **TurboQuant Fork**: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) - Walsh-Hadamard Transform KV compression and PolarQuant codecs.
- **RDNA Boosts**: [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) by Stew Forster - AMD RDNA-specific kernels, WMMA acceleration, and MoE optimizations.
- **ROCmFPX**: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) - ROCm FP4/FPx quantization formats and agent recipes.
- **Unsloth AI**: [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) - Shape-aware graph keying and runtime optimizations.
- **DFlash**: Upstream PR #27816 & PR #28000 by the DFlash contributors.

---

## Disclaimer

This is an experimental branch (`experiment/rdna-boosts`) consolidating bleeding-edge ROCm features, custom quantizations, and low-level kernel optimizations. While all test suites (`test-turbo-quant`, `test-quantize-fns`, `test-backend-ops`) pass with 100% correctness on verified hardware (including AMD Radeon RX 9060 XT gfx1200), features may evolve as upstream ggml specifications update.

---

## License

This project is licensed under the [MIT License](LICENSE).
