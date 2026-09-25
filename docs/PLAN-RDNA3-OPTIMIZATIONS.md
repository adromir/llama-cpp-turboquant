# Implementation Plan: HIP P2P AllReduce, DFlash2 Tensor Split Fix & Channels-Major SSM Conv

This document tracks the integration of three high-value optimizations and fixes from [nasone32/llama.cpp-RDNA3-7900xtx-opt](https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt) into our `experiment/rdna-boosts` branch.

---

## Technical Summary

1. **Two-GPU HIP AllReduce & Direct P2P (Without RCCL)**:
   - Bypasses host staging and RCCL by enabling bidirectional `cudaMemcpyPeerAsync` direct P2P transfers across PCIe for dual-GPU AMD setups.
   - Maps HIP pinned host memory allocators in `vendors/hip.h` and unguards `allreduce.cu` for `GGML_USE_HIP`.
2. **DFlash2 Speculative Tensor Split Fix**:
   - Replicates `output.weight` and `output.bias` (`GGML_BACKEND_SPLIT_AXIS_MIRRORED`) under tensor parallelism (`-ts`) when a DFlash2 / DSpark drafter ranks the full vocabulary in-graph without its own `lm_head`.
3. **Channels-Major SSM Conv for Gated-Delta-Net**:
   - Eliminates the expensive tensor transpose and contiguous copy (`CONT` kernel) of `qkv_mixed` in Gated-Delta-Net (GDN) models (Qwen 3.5, Qwen 3.8 Flash, Qwen 4, etc.) by teaching `ggml_ssm_conv` to accept channels-major tensors directly.

---

## Detailed Task List & Checkpoints

### Phase 1: Two-GPU HIP AllReduce & Direct P2P
- [x] **vendors/hip.h**:
  - Add macro mappings for `cudaHostAlloc` -> `hipHostMalloc`, `cudaHostAllocMapped` -> `hipHostMallocMapped`, `cudaHostAllocPortable` -> `hipHostMallocPortable`, `cudaHostGetDevicePointer` -> `hipHostGetDevicePointer`.
- [x] **allreduce.cu**:
  - Un-guard HIP at top and bottom: `#if !defined(GGML_USE_MUSA)` with `__builtin_amdgcn_s_sleep` polling and CUDA-only CC checks.
  - Remove redundant `allreduce-hip.cu` to eliminate duplicate symbols and unify allreduce across CUDA and HIP.
  - Add `p2p_stream`, `p2p_done`, `p2p_issuer`, and `p2p_enabled` to `struct ggml_cuda_ar_pipeline`.
  - In `ggml_cuda_ar_pipeline_init`: Check bidirectional peer access via `cudaDeviceCanAccessPeer` and enable peer access via `cudaDeviceEnablePeerAccess` when `n_devices == 2`. Allocate `p2p_stream` and `p2p_done` events.
  - In `ggml_cuda_ar_pipeline_free`: Clean up `p2p_done` events and `p2p_stream`.
  - Implement `ggml_cuda_ar_allreduce_p2p_impl<T_src, T_dst>` performing `cudaMemcpyPeerAsync` and local reduction with `ggml_cuda_ar_add_kernel`.
  - In `ggml_cuda_ar_allreduce_copy_outer`: Dispatch to `ggml_cuda_ar_allreduce_p2p_impl` whenever `p->p2p_enabled == true`.

### Phase 2: DFlash2 Speculative Tensor Split Fix
- [x] **include/llama.h**:
  - Add `bool output_replicated;` to `struct llama_model_params`.
- [x] **src/llama-model.h**:
  - Add `bool output_replicated = false;` to `struct llama_model`.
- [x] **src/llama-model.cpp**:
  - In `llama_meta_device_get_split_state`: Replicate `output.weight` and `output.bias` with `GGML_BACKEND_SPLIT_AXIS_MIRRORED` if `is_dsv4 || ud->model->output_replicated`.
  - In `llama_model::llama_model`: Propagate `output_replicated = params.output_replicated;`.
  - In `llama_model_default_params`: Default `output_replicated = false`.
- [x] **src/models/dflash.cpp**:
  - In `llama_model_dflash::load_arch_tensors`: Set `output_replicated = true` when `params.split_mode == LLAMA_SPLIT_MODE_TENSOR && (selector_meta || markov_meta)`.
- [x] **common/speculative.h**:
  - Declare `bool common_speculative_draft_ranks_full_output(const std::string & path);`.
- [x] **common/speculative.cpp**:
  - Implement `common_speculative_draft_ranks_full_output`: Inspect draft GGUF metadata for architecture `dflash` with `selector_hidden.weight` or `markov_w1.weight` lacking `output.weight`.
- [x] **common/common.cpp**:
  - In `common_init_result::common_init_result`: Automatically set `mparams.output_replicated = true` if `params.split_mode == LLAMA_SPLIT_MODE_TENSOR` and `common_speculative_draft_ranks_full_output(...)` is true.

### Phase 3: Qwen Flash / GDN Channels-Major SSM Conv
- [x] **ggml/include/ggml.h**:
  - Define `enum ggml_ssm_conv_layout { GGML_SSM_CONV_LAYOUT_TIME_MAJOR = 0, GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR = 1 };`.
  - Declare `ggml_ssm_conv_channels_major` and `ggml_ssm_conv_get_layout`.
- [x] **ggml/src/ggml.c**:
  - Refactor `ggml_ssm_conv` into `ggml_ssm_conv_impl(..., layout)` and set `op_params[0] = layout`.
  - Implement `ggml_ssm_conv` (time-major default) and `ggml_ssm_conv_channels_major`.
  - Implement `ggml_ssm_conv_get_layout`.
- [x] **ggml/src/ggml-cpu/ops.cpp**:
  - In `ggml_compute_forward_ssm_conv_f32`: Read layout via `ggml_ssm_conv_get_layout`. When channels-major, swap stride indexing between channels (`ne[0]`) and time (`ne[1]`).
- [x] **ggml/src/ggml-cuda/ssm-conv.cu**:
  - Add `bool channels_major` template parameter to `ssm_conv_f32` and `ssm_conv_long_token_f32`.
  - When `channels_major` is true, perform coalesced reads over consecutive channel addresses.
  - In `ggml_cuda_op_ssm_conv`: Extract layout and instantiate `<true, true>`, `<true, false>`, `<false, true>`, `<false, false>` kernels.
- [x] **Non-CUDA Backends**:
  - In `ggml-metal-device.m`, `ggml-vulkan.cpp`, `ggml-sycl.cpp`, `ggml-opencl.cpp`, `ggml-webgpu.cpp`, `ggml-hexagon.cpp`: Ensure `supports_op` verifies `ggml_ssm_conv_get_layout(op) == GGML_SSM_CONV_LAYOUT_TIME_MAJOR`.
  - In `ggml-backend-meta.cpp`: Relax assertion in `ggml_backend_meta_get_split_state` for reshape operations.
- [x] **Model Graphs**:
  - `src/models/delta-net-base.cpp`: In `build_conv_state`, reshape `conv_states` as `[conv_channels, conv_kernel_size - 1, n_seqs]`, prepend via `ggml_concat(..., 1)` along the time axis, drop `ggml_transpose` and `ggml_cont`.
  - `src/models/qwen35.cpp`: Call `ggml_ssm_conv_channels_major`.
  - `src/models/qwen35moe.cpp`: Call `ggml_ssm_conv_channels_major`.
  - `src/models/qwen3next.cpp`: Call `ggml_ssm_conv_channels_major`.
  - `src/models/qwen4exp.cpp`: Update `build_layer_attn_linear` and `build_conv_state_at` with `channels_major = true`.
- [x] **Tests**:
  - Add channels-major test cases to `test_ssm_conv` in `tests/test-backend-ops.cpp`.

---

## Verification Plan

### Automated Tests
- [x] Run `test-backend-ops -o SSM_CONV` comparing CPU reference and GPU kernels on both time-major and channels-major paths (90/90 passed on RX 9060 XT).
- [x] Run `test-turbo-quant` to ensure WHT and TurboQuant operations remain intact (100% roundtrip passed).
- [x] Run `test-quantize-fns` (50/50 quantization formats passed).
- [x] Run `test-backend-ops -o MUL_MAT -p type_a=tq4_1s` (149/149 passed) and `FLASH_ATTN_EXT -p type_V=turbo3` (960/960 passed).
- [x] Verify clean compilation without warnings on MSVC + clang / HIP compiler.
