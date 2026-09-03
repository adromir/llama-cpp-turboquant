<#
.SYNOPSIS
    Requantize an existing K-quant or Q8 GGUF model into ROCmFPX presets on Windows.

.DESCRIPTION
    Native PowerShell script to requantize GGUF models (e.g. Q4_K_M, Q6_K, Q8_0) to ROCmFPX presets
    without requiring the original BF16/F16 model source.
    Includes automated source quality ladder analysis and preset recommendations.

.PARAMETER Src
    Path to input quantized GGUF file (e.g. Q4_K_M, Q6_K, Q8_0).

.PARAMETER Out
    Path for output ROCmFPX GGUF file.

.PARAMETER Preset
    Target preset: Q3_0_ROCMFPX | Q6_0_ROCMFPX | Q8_0_ROCMFPX | *_AGENT variants (default: Q3_0_ROCMFPX).

.PARAMETER PresetForce
    Bypass automated preset recommendations (e.g. forcing Q3 from Q8 sources).

.PARAMETER Imatrix
    Optional importance matrix file for better Q3/Q6 quantization from Q8 sources.

.PARAMETER TensorTypeFile
    Optional tensor override file for experimental policies.

.PARAMETER AllowRequantize
    Allow requantizing tensors that are already quantized (default: true).

.PARAMETER DryRun
    Print planned command without writing file.

.PARAMETER NThreads
    Optional thread count for llama-quantize.

.PARAMETER QuantizeBin
    Path to llama-quantize.exe binary. Auto-discovered if omitted.

.EXAMPLE
    .\scripts\quantize-rocmfpx-from-kquant.ps1 -Src model-Q4_K_M.gguf -Out model-Q3_0_ROCMFPX.gguf

.EXAMPLE
    .\scripts\quantize-rocmfpx-from-kquant.ps1 -Src model-Q8_0.gguf -Out model-Q6_0_ROCMFPX.gguf -Imatrix imatrix.gguf
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [Alias("Source", "InputFile")]
    [string]$Src = $env:SRC,

    [Parameter(Position = 1)]
    [Alias("Output", "OutputFile")]
    [string]$Out = $env:OUT,

    [Parameter()]
    [string]$Preset = $(if ($env:PRESET) { $env:PRESET } else { "Q3_0_ROCMFPX" }),

    [Parameter()]
    [switch]$PresetForce = ($env:PRESET_FORCE -ne $null -and $env:PRESET_FORCE -ne ""),

    [Parameter()]
    [string]$Imatrix = $env:IMATRIX,

    [Parameter()]
    [string]$TensorTypeFile = $env:TENSOR_TYPE_FILE,

    [Parameter()]
    [switch]$AllowRequantize = ($env:ALLOW_REQUANTIZE -ne "0"),

    [Parameter()]
    [switch]$DryRun = ($env:DRY_RUN -eq "1"),

    [Parameter()]
    [int]$NThreads = $(if ($env:NTHREADS) { [int]$env:NTHREADS } else { 0 }),

    [Parameter()]
    [string]$QuantizeBin = $env:QUANTIZE_BIN
)

$ErrorActionPreference = "Stop"

# Enable unified memory by default for ROCm to avoid VRAM allocation failures
if (-not $env:GGML_HIP_ENABLE_UNIFIED_MEMORY) {
    $env:GGML_HIP_ENABLE_UNIFIED_MEMORY = "1"
}

# Auto-discover llama-quantize.exe
if (-not $QuantizeBin) {
    $candidates = @(
        "$PSScriptRoot\..\build\bin\llama-quantize.exe",
        "$PSScriptRoot\..\build\bin\Release\llama-quantize.exe",
        "$PSScriptRoot\..\build-strix-rocmfp4\bin\llama-quantize.exe",
        "$PSScriptRoot\..\build\bin\llama-quantize"
    )
    foreach ($cand in $candidates) {
        if (Test-Path $cand) {
            $QuantizeBin = (Resolve-Path $cand).Path
            break
        }
    }
    if (-not $QuantizeBin) {
        $cmd = Get-Command "llama-quantize.exe" -ErrorAction SilentlyContinue
        if ($cmd) {
            $QuantizeBin = $cmd.Source
        }
    }
}

if (-not $Src -or -not $Out) {
    Write-Error "Both -Src (input GGUF) and -Out (output GGUF) are required."
    exit 2
}

if (-not $QuantizeBin -or -not (Test-Path $QuantizeBin)) {
    Write-Error "Could not find llama-quantize binary. Build it with .\build.ps1 or specify -QuantizeBin."
    exit 1
}

if (-not (Test-Path $Src)) {
    Write-Error "Source file does not exist: $Src"
    exit 1
}

if ($Imatrix -and -not (Test-Path $Imatrix)) {
    Write-Error "Imatrix file does not exist: $Imatrix"
    exit 1
}

if ($TensorTypeFile -and -not (Test-Path $TensorTypeFile)) {
    Write-Error "Tensor type file does not exist: $TensorTypeFile"
    exit 1
}

# Quality ladder analysis and warnings
$srcBase = [System.IO.Path]::GetFileName($Src)
$warn = $null

if ($srcBase -match "rocmfpx" -and $Preset -match "ROCMFPX") {
    $warn = "Source is already ROCmFPX; double requant often fails coherency - prefer Q4_K_M/Q6_K/Q8_0 stock quants"
} elseif ($srcBase -match "Q3_K") {
    $warn = "Q3_K source is below practical floor for Q3_0_ROCMFPX; expect coherency failures"
}

# Recommend Q6_0_ROCMFPX if source is Q8 and target was Q3
if (-not $PresetForce) {
    if ($srcBase -match "Q8" -and $Preset -like "Q3_0_ROCMFPX*") {
        Write-Host "NOTE: For Q8-class source targeting Q3, recommended preset is Q6_0_ROCMFPX (pass -PresetForce to override)." -ForegroundColor Yellow
        $Preset = "Q6_0_ROCMFPX"
    }
}

$quantArgs = @()
if ($AllowRequantize) {
    $quantArgs += "--allow-requantize"
}
if ($DryRun) {
    $quantArgs += "--dry-run"
}
if ($Imatrix) {
    $quantArgs += @("--imatrix", $Imatrix)
}
if ($TensorTypeFile) {
    $quantArgs += @("--tensor-type-file", $TensorTypeFile)
}

$quantArgs += @($Src, $Out, $Preset)
if ($NThreads -gt 0) {
    $quantArgs += $NThreads.ToString()
}

$outDir = [System.IO.Path]::GetDirectoryName($Out)
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

Write-Host "========== ROCmFPX Requantize from K-Quant ==========" -ForegroundColor Cyan
Write-Host "Source:           $Src"
Write-Host "Output:           $Out"
Write-Host "Preset:           $Preset"
if ($warn)           { Write-Host "WARNING:          $warn" -ForegroundColor Yellow }
if ($Imatrix)        { Write-Host "Imatrix:          $Imatrix" }
if ($TensorTypeFile) { Write-Host "TensorTypeFile:   $TensorTypeFile" }
Write-Host "Quantize Binary:  $QuantizeBin"
Write-Host "======================================================" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "[DRY RUN] Executing dry run check..." -ForegroundColor Yellow
}

& $QuantizeBin @quantArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "llama-quantize exited with error code $LASTEXITCODE"
    exit $LASTEXITCODE
}

if (-not $DryRun) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Out)
    $ext = [System.IO.Path]::GetExtension($Out)
    $searchDir = if ($outDir) { $outDir } else { "." }
    $shardPattern = "$stem-*-of-*$ext"
    $shards = Get-ChildItem -Path $searchDir -Filter $shardPattern -ErrorAction SilentlyContinue

    $fileList = @()
    if ($shards -and $shards.Count -gt 0) {
        foreach ($f in $shards) {
            $fileList += [PSCustomObject]@{
                Path = $f.FullName
                Bytes = $f.Length
            }
        }
    } elseif (Test-Path $Out) {
        $f = Get-Item $Out
        $fileList += [PSCustomObject]@{
            Path = $f.FullName
            Bytes = $f.Length
        }
    }

    $summary = [PSCustomObject]@{
        Status          = "pass"
        Source          = $Src
        Output          = $Out
        Preset          = $Preset
        Warning         = if ($warn) { $warn } else { $null }
        AllowRequantize = [bool]$AllowRequantize
        Imatrix         = if ($Imatrix) { $Imatrix } else { $null }
        TensorTypeFile  = if ($TensorTypeFile) { $TensorTypeFile } else { $null }
        Files           = $fileList
    }

    $json = $summary | ConvertTo-Json -Depth 4
    Write-Output $json
}
