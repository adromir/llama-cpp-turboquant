# Adaptive KV Streaming & Tiered-Memory Concept

## Overview

This document describes the architecture, mathematical mechanisms, and future integration plan for **Adaptive KV Cache Streaming** in this fork.

The concept is based on research and implementation by Raymond Huang in [llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) (tracked locally in branch `raymond/adaptive-kv-stream`), combined with the high-ratio KV compression of **TurboQuant** and AMD ROCm/HIP hardware acceleration.

---

## 1. Problem Statement: The Ultra-Long-Context VRAM Wall

When serving ultra-long contexts (e.g. 128k, 256k, or 512k+ tokens) on consumer or prosumer GPUs with limited VRAM (16 GB - 24 GB), the KV cache size often exceeds total device memory, even if model weights are heavily quantized.

Traditional fallbacks have severe drawbacks:

1. **Unified Memory (CUDA UVM / ROCm Managed Memory):**
   - Relying on driver-level page migration causes thrashing.
   - Page faults on the GPU execution path block wavefronts, dropping decode throughput to < 0.1 tokens/s.
2. **Context Truncation / Sliding Window:**
   - Drops early tokens, losing long-range document context.
3. **KV Sparsification / Token Dropping:**
   - May degrade reasoning quality or recall accuracy for "needle-in-a-haystack" queries.

---

## 2. Core Architecture: Adaptive Block-Granular KV Streaming

Adaptive KV Streaming solves the capacity limitation through deterministic, block-granular asynchronous staging from host system RAM (CPU memory) into a bounded GPU VRAM pool.

```
+-------------------------------------------------------------------------+
|                        System RAM (Host Pinned)                         |
|         Authoritative KV Cache (Full Context, e.g. 262,144 tokens)      |
+-------------------------------------------------------------------------+
                                     |
                                     | PCIe 4.0/5.0 (Async Stream Transfer)
                                     v
+-------------------------------------------------------------------------+
|                     GPU VRAM Pool (--kv-stream-stage-mib N)             |
|                                                                         |
|  +-----------------------------------+  +----------------------------+  |
|  |       Resident Pages Cache        |  |    Transfer Staging Ring   |  |
|  |  (Recent tokens kept in fast VRAM)|  | (Prefetched layer slots)   |  |
|  +-----------------------------------+  +----------------------------+  |
+-------------------------------------------------------------------------+
                                     |
                                     v
                  [ Flash Attention with Online Softmax ]
```

### Key Components

### 1. Pinned Host Memory Allocation
- The authoritative KV cache tensors reside in page-locked (pinned) CPU system memory via `cudaHostAlloc` / `hipHostMalloc`.
- This ensures maximum direct-memory-access (DMA) throughput over PCIe without host-side paging delays.

### 2. Bounded GPU Pool (`--kv-stream-stage-mib N`)
- A fixed amount of GPU VRAM (e.g. 2304 MiB) is reserved for the streaming runtime.
- The pool is divided into two areas:
  - **Resident Cache:** Holds the newest / most frequently referenced KV pages (e.g. recent context tail) to avoid streaming them repeatedly.
  - **Transfer Ring:** A circular buffer of staging slots used to stream older KV pages from host RAM on-demand.

### 3. Asynchronous Layer-Ahead Prefetching
- Staging transfers run on a dedicated asynchronous copy stream (`hipStream_t` / `cudaStream_t`) synchronized by hardware events (`hipEvent_t` / `cudaEvent_t`).
- **Pipelined Execution:** While the compute stream evaluates Attention for layer $L$, the copy engine prefetches the required KV blocks for layer $L+1$ from host RAM into free slots in the GPU transfer ring.

### 4. Partial FlashAttention & Exact Online Softmax Merging
When the KV cache for a layer is split into multiple streamed blocks, FlashAttention cannot compute the entire sequence in a single kernel pass. Instead, it evaluates partial blocks and merges them using numerically exact online softmax reduction.

For each block $i$, the kernel outputs:
- Partial unnormalized numerator $O_i$
- Running row maximum $m_i$
- Running row sum of exponentials $l_i$

Two blocks ($A$ and $B$) are merged via:

$$m_{new} = \max(m_A, m_B)$$

$$l_{new} = l_A \cdot e^{m_A - m_{new}} + l_B \cdot e^{m_B - m_{new}}$$

$$O_{new} = O_A \cdot e^{m_A - m_{new}} + O_B \cdot e^{m_B - m_{new}}$$

Final attention output:

$$V_{out} = \frac{O_{new}}{l_{new}}$$

This provides exact mathematical equivalence to standard FlashAttention without any approximation or precision loss.

### 5. Adaptive Partitioning & Hysteresis
- The scheduler tracks transfer deadline misses (when compute must wait for a PCIe transfer to complete).
- If transfers bottleneck compute, the runtime dynamically reallocates pages from the Resident Cache to enlarge the Transfer Ring.
- Hysteresis thresholds prevent thrashing between prefill and decode phases.

---

## 3. Synergy with TurboQuant: "Tiered TurboQuant"

Adaptive KV Streaming in Raymond's repository was built for standard types (`q8_0`, `q4_0`, `f16`). However, the primary physical limit of PCIe streaming during decode is **PCIe bus bandwidth**:

| Cache Type | Bytes/Token/Head (Dim 128) | Bandwidth Demand at 128k Ctx | PCIe 4.0 x16 Saturation |
| :--- | :--- | :--- | :--- |
| `f16` | 256 bytes | ~32.8 MB / layer / token | High bottleneck |
| `q8_0` | 132 bytes | ~16.9 MB / layer / token | Moderate bottleneck |
| `turbo3_0` (Ours) | **52 bytes** | **~6.6 MB / layer / token** | **Low (2.5x faster)** |
| `turbo2_0` (Ours) | **32 bytes** | **~4.1 MB / layer / token** | **Minimal (4.1x faster)** |

### The Tiered TurboQuant Vision:
1. **Compress in Host RAM:** Host memory stores the KV cache in `turbo2_0` (2 bits) or `turbo3_0` (3.25 bits) after Walsh-Hadamard transformation (`GGML_OP_TURBO_WHT`).
2. **Transfer Compressed Blocks:** The DMA copy stream transfers 4x smaller payloads across the PCIe bus, virtually eliminating the transfer deadline misses.
3. **Dequantize & Compute on GPU:** The GPU staging ring receives the compressed blocks directly and feeds them to the native TurboQuant FlashAttention kernels.

---

## 4. Porting Roadmap & Architectural Requirements

To integrate this concept into our fork without regressions, the following steps are required:

### Phase 1: Decouple from CUDA-Only APIs
- Replace hardcoded CUDA calls in `src/llama-kv-stream-plan.cpp` and `src/llama-kv-stream-config.cpp` with backend-neutral interfaces or HIP equivalents (`hipMalloc`, `hipMemcpyAsync`, `hipEventCreateWithFlags`).
- Remove the backend check in `src/llama-kv-cache.cpp` that currently throws `"block KV streaming requires the CUDA backend"`.

### Phase 2: ROCm / HIP Partial FlashAttention Kernels
- Port the partial vector and MMA FlashAttention dispatch logic from `ggml/src/ggml-cuda/fattn.cu` to HIP.
- Ensure compatibility with AMD RDNA Wave32 execution (`GGML_AMDGPU_WAVES_PER_EU(4, 8)`) and WMMA/V_DOT intrinsics.
- Integrate online softmax merging directly into the HIP attention epilogue.

### Phase 3: TurboQuant Integration
- Extend `llama_kv_stream_type_capabilities` to recognize `GGML_TYPE_TURBO2_0`, `GGML_TYPE_TURBO3_0`, and `GGML_TYPE_TURBO4_0`.
- Ensure WHT coordinate domain consistency across partial chunks.
- Connect to non-temporal streaming loads (`ggml_cuda_ldcs`) on RDNA to avoid L2 cache pollution when reading staged ring buffers.

### Phase 4: Verification & Benchmarking
- Add test suites modeled after `tests/test-kv-stream-plan.cpp` and `tests/test-kv-stream-softmax.cpp`.
- Verify numerical parity against non-streamed `test-backend-ops -o FLASH_ATTN_EXT`.
- Benchmark tokens/s at 128k, 256k, and 512k context lengths on AMD Radeon RX 9000 series hardware.

---

## 5. Reference File Mapping

| Component | Upstream / Reference Location (`raymond/adaptive-kv-stream`) |
| :--- | :--- |
| Stream Configuration | `src/llama-kv-stream-config.h`, `src/llama-kv-stream-config.cpp` |
| Planner & Residency Engine | `src/llama-kv-stream-plan.h`, `src/llama-kv-stream-plan.cpp` |
| Softmax Merge Math | `src/llama-kv-stream-softmax.h`, `src/llama-kv-stream-softmax.cpp` |
| Runtime Interface | `ggml/include/ggml-cuda.h` |
| CUDA Staging & Partial FA | `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/ggml-cuda.cu` |
| Span Tuner | `ggml/src/ggml-cuda/kv-stream-span-tuner.h` |
| Benchmark Suite | `benchmarks/benchmark_kv_stream.py`, `tools/kv-stream-bench/` |
