# Langfristiger ROCm 10 Optimierungs- und Integrationsplan

**Version:** 1.0  
**Status:** In Vorbereitung  
**Zielsystem:** Windows 11 / Linux (ROCm 10.0.0 "TheRock")  
**Zielhardware:** AMD RDNA4 (Radeon RX 9060 XT / gfx1200 / gfx1201), RDNA3 (RX 7000 / gfx1100), RDNA3.5 (Strix / gfx1151)  
**Autor:** Antigravity AI & Adromir  

---

## 1. Executive Summary & Ausgangslage

Mit dem Release von **ROCm 10.0.0** Ende August 2026 ("TheRock") hat AMD den gesamten Compute-Stack grundlegend modernisiert. Auf dem Entwicklungs-System steht die vollständige ROCm 10 Suite (`C:\TheRock\build`) mit aktuellem LLVM/Clang 23 Compiler, `libhipblaslt`, `ck`/`ck_tile` und `rocwmma` zur Verfügung.

Upstream-`llama.cpp` bindet ROCm bisher primär über eine generische HIP-Übersetzungsschicht (`vendors/hip.h`) ein, die auf Legacy-`hipBLAS` (altes `rocBLAS`) aufsetzt. Dadurch bleiben erhebliche Hardwareressourcen von RDNA3/RDNA4 ungenutzt.

Dieser Plan definiert eine schrittweise, risikofreie Roadmap, um alle relevanten ROCm 10 Optimierungen nachhaltig in die Codebasis zu integrieren.

---

## 2. Übersicht: Sortierung nach Schwierigkeit und Aufwand

| Phase | Optimierungsbereich | Schwierigkeit | Aufwand | Nutzen / Impact | Primäre Zielkomponente |
|---|---|---|---|---|---|
| **Phase 1** | **RDNA4 / RDNA3.5 WMMA Compiler-Intrinsics** | Gering | Gering bis Mittel (1-2 Tage) | Sehr hoch (Sofortiger GEMV/GEMM-Gewinn für FP4) | `ggml-cuda/vecdotq.cuh`, `mmvq-tq.cu` |
| **Phase 2** | **Natives `hipBLASLt`-Backend mit Workspace** | Mittel | Mittel (2-3 Tage) | Extrem hoch (Prefill-Durchsatz & Batched GEMM) | `vendors/hip.h`, `ggml-cuda.cu`, `ggml-hip/CMakeLists.txt` |
| **Phase 3** | **HIP-Graph-Stabilisierung im Decode-Loop** | Mittel | Mittel (2-3 Tage) | Hoch (Minimiert CPU/GPU-Launch-Latenz) | `ggml-cuda.cu`, `llama-context.cpp` |
| **Phase 4** | **TunableOp / GEMM Shape-Cache** | Mittel bis Hoch | Mittel (3-4 Tage) | Hoch (10-25% Plus bei exotischen Kontexten) | `vendors/hip.h`, `ggml-cuda.cu` |
| **Phase 5** | **Composable Kernel (CK) FlashAttention Kachelung** | Hoch | Hoch (4-6 Tage) | Hoch (Stabilität & Speed bei D=256 / langen Kontexten) | `ggml-cuda/fattn-wmma.cu` |

---

## 3. Detaillierte Phasenplanung

---

### Phase 1: RDNA4 & RDNA3.5 WMMA Compiler-Intrinsics
* **Schwierigkeitsgrad:** Gering  
* **Aufwand:** Gering bis Mittel  
* **Ziel:** Ausschöpfung der im ROCm 10 Clang 23 Compiler nativ verfügbaren Intrinsics für Matrixkerne (WMMA).

#### Hintergrund
In RDNA3 und RDNA4 (`gfx1100`, `gfx1151`, `gfx1200`, `gfx1201`) verfügt die Hardware über dedizierte WMMA-Instruktionen. Älterer Code nutzt Inline-Assembly oder dp4a-Emulationen, was den LLVM-Register-Allokator behindert. Der Clang 23 Compiler in ROCm 10 liefert standardisierte Builtins:
* `__builtin_amdgcn_wmma_f32_16x16x16_f16`
* `__builtin_amdgcn_wmma_f32_16x16x16_bf16`
* `__builtin_amdgcn_wmma_i32_16x16x16_iu4` / `_iu8` (RDNA4 Sub-Byte Support)

#### Betroffene Dateien
* [vecdotq.cuh](file:///e:/llama-cpp-turboquant/ggml/src/ggml-cuda/vecdotq.cuh)
* [mmvq-tq.cu](file:///e:/llama-cpp-turboquant/ggml/src/ggml-cuda/mmvq-tq.cu)
* [common.cuh](file:///e:/llama-cpp-turboquant/ggml/src/ggml-cuda/common.cuh)

#### Implementierungsschritte
1. Feature-Detection Makros in `common.cuh` für ROCm 10 Clang Versionen (`__clang_major__ >= 19`) ergänzen.
2. In `vecdotq.cuh` und `mmvq-tq.cu` prüfen, ob RDNA4 (`__HIP_DEVICE_COMPILE__` mit gfx12) aktiv ist, und native Intrinsics für Bit-Packing und Multiplikation schalten.
3. Bereinigung von überflüssigen Permutations-Ketten bei `tq4_1s` und `Q4_0_ROCMFP4_FAST`.

#### Verifikation
```powershell
# Unit-Tests fuer Quantisierung und mathematische Korrektheit
.\build\bin\test-quantize-fns.exe
.\build\bin\test-backend-ops.exe -o MUL_MAT -p "type_a=tq4_1s"
.\build\bin\test-backend-ops.exe -o MUL_MAT -p "type_a=q4_0"
```

---

### Phase 2: Natives `hipBLASLt`-Backend mit Heuristik & Workspace
* **Schwierigkeitsgrad:** Mittel  
* **Aufwand:** Mittel  
* **Ziel:** Vollständige Ablösung von veraltetem `hipblasGemmEx` durch modernes `hipblasLtMatmul`.

#### Hintergrund
`hipBLAS` (Wrapper um altes `rocBLAS`) wählt Kernel rein statisch aus und unterstützt weder Epilogue-Fusion noch moderne TensileLt-Kacheln sauber. `hipBLASLt` ist AMDs Gegenstück zu NVIDIAs `cuBLASLt` und ist in ROCm 10 hochgradig performant.

#### Betroffene Dateien
* [CMakeLists.txt (ggml-hip)](file:///e:/llama-cpp-turboquant/ggml/src/ggml-hip/CMakeLists.txt)
* [CMakeLists.txt (root)](file:///e:/llama-cpp-turboquant/ggml/CMakeLists.txt)
* [vendors/hip.h](file:///e:/llama-cpp-turboquant/ggml/src/ggml-cuda/vendors/hip.h)
* [ggml-cuda.cu](file:///e:/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu)

#### Implementierungsschritte
1. **CMake:** Option `GGML_HIP_BLASLT` hinzufügen.
   ```cmake
   find_package(hipblaslt REQUIRED)
   target_link_libraries(ggml-hip PRIVATE roc::hipblaslt)
   add_compile_definitions(GGML_USE_HIPBLASLT)
   ```
2. **Context & Workspace:**
   * Initialisierung des `hipblasLtHandle_t` im Device-Context (`ggml_cuda_context`).
   * Zuweisung eines persistenten Scratch-Puffers (z. B. 32 MB bis 64 MB VRAM) für `hipBLASLt`-Workspace.
3. **Matmul Dispatcher (`ggml_cuda_mul_mat_cublas`):**
   * Wrapper-Funktion `ggml_hipblaslt_matmul(...)` implementieren.
   * Aufruf von `hipblasLtMatmulPreferenceCreate`, Zuweisung des Workspaces.
   * Abfrage von `hipblasLtMatmulAlgoGetHeuristic`.
   * **Wichtig - Fail-Safe Fallback:** Liefert die Heuristik keinen Treffer (z. B. bei ungeraden Matrixmaßen), fällt der Dispatcher nahtlos und geräuschlos auf `hipblasGemmEx` zurück.

#### Verifikation
```powershell
# Alle Backend-Ops muessen gruen bleiben
.\build\bin\test-backend-ops.exe -o MUL_MAT
# Performance-Messung des Prompt-Processing (Prefill -pp)
.\build\bin\llama-bench.exe -m model.gguf -p 512,1024,2048 -n 0
```

---

### Phase 3: HIP-Graph Stabilisierung im Decode-Loop
* **Schwierigkeitsgrad:** Mittel  
* **Aufwand:** Mittel  
* **Ziel:** Beseitigung von CPU-Kernel-Launch-Jitter bei der Token-für-Token-Generierung auf Windows.

#### Hintergrund
Beim Single-Token-Decode (Batch = 1) dominiert oft die Latenz der CPU-Treiber-Submissions (besonders unter Windows WDDM). Mit HIP-Graphs wird der gesamte Ausführungsgraph einmalig auf der GPU aufgezeichnet (`hipStreamBeginCapture`) und danach mit einem einzigen Aufruf (`hipGraphLaunch`) ausgeführt.

#### Betroffene Dateien
* [ggml-cuda.cu](file:///e:/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu)
* [common.cuh](file:///e:/llama-cpp-turboquant/ggml/src/ggml-cuda/common.cuh)

#### Implementierungsschritte
1. Analyse der Abbruch-Bedingungen von `hipStreamEndCapture` unter ROCm 10.
2. Sicherstellen, dass alle Puffer im Graph feste Adressen besitzen (kein Re-Allocating während des Decodes).
3. Absichern des MoE- und TurboQuant-Routen gegen Re-Capturing.
4. Freigabe von `GGML_HIP_GRAPHS` standardmäßig für RDNA3 und RDNA4.

#### Verifikation
```powershell
# Validierung der Textausgabe und Tokens/s beim Decode
.\build\bin\llama-bench.exe -m model.gguf -p 0 -n 128 -t 1
```

---

### Phase 4: TunableOp & GEMM Shape-Cache
* **Schwierigkeitsgrad:** Mittel bis Hoch  
* **Aufwand:** Mittel  
* **Ziel:** Dynamisches Finden des perfekten GPU-Kernels für jedes Modell und jede Kontextlänge.

#### Hintergrund
Statt sich auf statische Heuristiken zu verlassen, profiliert ROCm 10 verschiedene Kernel-Implementierungen direkt auf der GPU. Für gegebene Matrix-Dimensionen $(M, N, K)$ wird die gemessene Bestzeit in einer internen Cache-Tabelle hinterlegt.

#### Betroffene Dateien
* [ggml-cuda.cu](file:///e:/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu)
* Neues Modul oder Erweiterung: `ggml-cuda/rocm-tuning.cuh`

#### Implementierungsschritte
1. Implementierung eines In-Memory Shape-Caches: `std::unordered_map<ShapeKey, hipblasLtMatmulAlgo_t>`.
2. Während des initialen Graph-Warmups testet `hipblasLtMatmulAlgoGetHeuristic` die Top-3-Kandidaten und speichert den schnellsten Treffer.
3. Optionale Persistierung der Tabelle in einer Datei (`.ggml-rocm-tuning.bin`), um erneutes Benchmarking beim nächsten Start zu vermeiden.

#### Verifikation
```powershell
# Wiederholte Benchmarks zur Bestaetigung des Speedups
.\build\bin\llama-bench.exe -m model.gguf -p 2048 -n 128 -r 5
```

---

### Phase 5: Composable Kernel (CK) FlashAttention Kachel- und Register-Schemata [ABGESCHLOSSEN]
* **Status:** Erfolgreich implementiert und verifiziert auf AMD Radeon RX 9060 XT (gfx1200).
* **Ziel:** Maximale FlashAttention-Performance und Vermeidung von Regressionen bei $D=128, 256$ (Gemma, Qwen, DeepSeek, etc.) auf RDNA3/RDNA4.

#### Hintergrund
Die AMD Composable Kernel (CK) Bibliothek bietet die fortschrittlichsten Aufmerksamkeits-Kernel für AMD-Architekturen.
**Strategie:** Wir portieren die erprobten Kachel- und Registerverteilungs-Muster (Tile-Shapes $kN0 = 64$, 4-Wave32 Workgroups mit 128 Threads zur 100% WGP-SIMD-Auslastung, non-temporale KV-Cache-Loads) direkt in `fattn-mma-f16.cuh`.

#### Betroffene Dateien
* [fattn-mma-f16.cuh](file:///e:/llama-cpp-turboquant/ggml/src/ggml-cuda/fattn-mma-f16.cuh)

#### Implementierung
1. Analyse der CK-Tile-Konfigurationen fuer gfx1100 und gfx1200 in `C:\TheRock\build\include\ck_tile`.
2. Dedizierte RDNA4-Konfiguration `ggml_cuda_fattn_mma_get_config_rdna4` und Aktualisierung von `ggml_cuda_fattn_mma_get_config_rdna` (fuer RDNA3) mit 128 Threads (4 Wave32 Warps) und `nbatch_fa = 64` fuer $D=128$ und $D=256$.
3. Non-temporale Streaming-Loads (`ggml_cuda_memcpy_streaming<16>`) beim Laden der KV-Cache-Zeilen in `flash_attn_ext_f16_load_tile` zur Vermeidung von L1/L2-Cache-Thrashing.
4. Alle Test-Gates (100% gruen):
   - `test-backend-ops.exe -o FLASH_ATTN_EXT -p "hsk=128"`: 6032/6032 bestanden.
   - `test-backend-ops.exe -o FLASH_ATTN_EXT -p "hsk=256"`: 1062/1062 bestanden.
   - `test-backend-ops.exe -o FLASH_ATTN_EXT -p "type_V=turbo3"`: 960/960 bestanden.
   - `test-turbo-quant.exe`: 100% bestanden.
   - `test-quantize-fns.exe`: 50/50 bestanden.
   - `llama-bench.exe` (Qwen 3.5 4B): Prefill > 3,725 tokens/s (`pp2048`), Decode 79.1 tokens/s.

---

## 4. Rollout-Strategie & Sicherheitsnetz

Um die bestehende Stabilität des `experiment/rdna-boosts`-Branches nicht zu gefährden:
1. **Phasenweises Vorgehen:** Jede Phase wird als separater, in sich geschlossener Meilenstein implementiert.
2. **Kompilier-Gates:** Keine Phase wird gemergt, bevor alle drei Haupt-Testsuiten (`test-quantize-fns`, `test-turbo-quant`, `test-backend-ops`) lokal auf der RX 9060 XT vollständig grün durchlaufen.
3. **Fail-Safe Fallbacks:** Jede neue ROCm 10 API erhält einen bedingungslosen Fallback auf die bisherige, bewährte Implementierung.
4. **Keine externen Zwang-Abhängigkeiten:** Der Build muss auch weiterhin auf Systemen ohne ROCm 10 (z. B. älteren ROCm 6/7 oder Standard-CUDA-Boxen) einwandfrei kompilieren.
