<#
.SYNOPSIS
    Quantize GGUF models to ROCmFPX and TurboQuant presets on Windows.

.DESCRIPTION
    Wrapper around llama-quantize for ROCmFP4, ROCmFPX, ROCmI4, and TurboQuant formats.
    Supports quantizing from BF16/F16 models or requantizing from existing K-quants (Q4_K_M, Q8_0).

.EXAMPLE
    .\scripts\quantize-rocmfpx.ps1 -Source "models\model-f16.gguf" -Output "models\model-rocmfp4.gguf" -Preset Q4_0_ROCMFP4_FAST

.EXAMPLE
    .\scripts\quantize-rocmfpx.ps1 -Source "models\model-Q8_0.gguf" -Output "models\model-rocmfp6.gguf" -Preset Q6_0_ROCMFPX -AllowRequantize
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Source,

    [Parameter(Mandatory=$true, Position=1)]
    [string]$Output,

    [Parameter(Position=2)]
    [ValidateSet(
        "Q4_0_ROCMFP4", "Q4_0_ROCMFP4_FAST", "Q4_0_ROCMFP4_COHERENT", "Q4_0_ROCMFP4_STRIX", "Q4_0_ROCMFP4_STRIX_LEAN",
        "Q3_0_ROCMFPX", "Q3_0_ROCMFPX_AGENT",
        "Q6_0_ROCMFPX", "Q6_0_ROCMFPX_AGENT", "Q6_0_ROCMFPX_LEAN",
        "Q8_0_ROCMFPX", "Q8_0_ROCMFPX_AGENT",
        "Q4_0_ROCMI4",
        "tq3_1s", "tq4_1s"
    )]
    [string]$Preset = "Q4_0_ROCMFP4_FAST",

    [Parameter()]
    [string]$Imatrix = "",

    [Parameter()]
    [string]$TensorTypeFile = "",

    [Parameter()]
    [switch]$AllowRequantize,

    [Parameter()]
    [int]$Threads = 0,

    [Parameter()]
    [string]$QuantizeBin = ""
)

$ErrorActionPreference = "Stop"

# 1. Locate llama-quantize executable
if (-not $QuantizeBin) {
    $Candidates = @(
        "$PSScriptRoot\..\build\bin\llama-quantize.exe",
        "$PSScriptRoot\..\build-rocm\bin\llama-quantize.exe",
        "$PSScriptRoot\..\build-strix-rocmfp4\bin\llama-quantize.exe",
        "llama-quantize.exe"
    )
    foreach ($cand in $Candidates) {
        if (Test-Path $cand) {
            $QuantizeBin = (Resolve-Path $cand).Path
            break
        }
    }
    if (-not $QuantizeBin) {
        $cmd = Get-Command "llama-quantize" -ErrorAction SilentlyContinue
        if ($cmd) { $QuantizeBin = $cmd.Source }
    }
}

if (-not $QuantizeBin -or -not (Test-Path $QuantizeBin)) {
    Write-Error "Could not find llama-quantize executable! Please build it or specify -QuantizeBin path."
    exit 1
}

# 2. Validate source file
if (-not (Test-Path $Source)) {
    Write-Error "Source file not found: $Source"
    exit 1
}

# 3. Create destination folder
$OutputDir = Split-Path -Parent $Output
if ($OutputDir -and -not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# 4. Construct arguments
$QuantArgs = @()
if ($AllowRequantize) {
    $QuantArgs += "--allow-requantize"
}
if ($Imatrix) {
    if (-not (Test-Path $Imatrix)) {
        Write-Error "Imatrix file not found: $Imatrix"
        exit 1
    }
    $QuantArgs += "--imatrix"
    $QuantArgs += $Imatrix
}
if ($TensorTypeFile) {
    if (-not (Test-Path $TensorTypeFile)) {
        Write-Error "Tensor type file not found: $TensorTypeFile"
        exit 1
    }
    $QuantArgs += "--tensor-type-file"
    $QuantArgs += $TensorTypeFile
}

$QuantArgs += $Source
$QuantArgs += $Output
$QuantArgs += $Preset

if ($Threads -gt 0) {
    $QuantArgs += $Threads.ToString()
}

Write-Host "=================================================="
Write-Host " ROCmFPX & TurboQuant Quantizer"
Write-Host "=================================================="
Write-Host "Binary:  $QuantizeBin"
Write-Host "Source:  $Source"
Write-Host "Output:  $Output"
Write-Host "Preset:  $Preset"
if ($Imatrix) { Write-Host "Imatrix: $Imatrix" }
Write-Host "=================================================="

& $QuantizeBin @QuantArgs

if ($LASTEXITCODE -eq 0 -and (Test-Path $Output)) {
    $sizeBytes = (Get-Item $Output).Length
    $sizeMB = [math]::Round($sizeBytes / 1MB, 2)
    Write-Host "`n[OK] Successfully created $Output ($sizeMB MB)" -ForegroundColor Green
} else {
    Write-Error "Quantization failed with exit code $LASTEXITCODE."
}
