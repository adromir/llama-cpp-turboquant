<#
.SYNOPSIS
    WPF Graphical Interface for llama.cpp Quantization (ROCmFPX, TurboQuant, Standard).

.DESCRIPTION
    Provides an interactive Windows GUI to select llama-quantize executable,
    input/output GGUF models, quantization presets, importance matrix, and options.
    Auto-detects llama-quantize.exe if located in the same directory as the script.
#>

param(
    [string]$InitialSource = "",
    [string]$InitialPreset = "Q4_0_ROCMFP4_FAST",
    [string]$InitialImatrix = "",
    [string]$InitialCalibration = ""
)

Set-StrictMode -Off

# Ensure WPF assemblies are loaded
Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase, System.Windows.Forms | Out-Null

# Force Software Rendering to prevent D3D/DirectX quota exhaustion (Win32Exception 1816) during heavy GPU/VRAM workloads
try {
    $roType = [Type]::GetType('System.Windows.Media.RenderOptions, PresentationCore')
    $rmType = [Type]::GetType('System.Windows.Interop.RenderMode, PresentationCore')
    if ($roType -and $rmType) {
        $roType.GetProperty('ProcessRenderMode').SetValue($null, [Enum]::Parse($rmType, 'SoftwareOnly'))
    }
} catch {}

# Define asynchronous background process runner with thread-safe queue.
# Pure C# event handlers prevent PowerShell 'no Runspace available on this thread' crashes on ThreadPool threads.
if (-not ('LlamaAsyncProcessRunner' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.Collections.Concurrent;

public class LlamaAsyncProcessRunner {
    public Process Process { get; private set; }
    public ConcurrentQueue<string> OutputLines = new ConcurrentQueue<string>();
    public volatile bool HasExited = false;
    public int ExitCode = -1;

    public bool Start(string exe, string args) {
        return Start(exe, args, null);
    }

    public bool Start(string exe, string args, string workingDir) {
        Process = new Process();
        Process.StartInfo.FileName = exe;
        Process.StartInfo.Arguments = args;
        Process.StartInfo.UseShellExecute = false;
        Process.StartInfo.RedirectStandardOutput = true;
        Process.StartInfo.RedirectStandardError = true;
        Process.StartInfo.CreateNoWindow = true;
        Process.EnableRaisingEvents = true;

        if (!string.IsNullOrEmpty(workingDir)) {
            Process.StartInfo.WorkingDirectory = workingDir;
        } else {
            try {
                string dir = System.IO.Path.GetDirectoryName(exe);
                if (!string.IsNullOrEmpty(dir)) {
                    Process.StartInfo.WorkingDirectory = dir;
                }
            } catch {}
        }

        Process.OutputDataReceived += (s, e) => {
            if (e.Data != null) OutputLines.Enqueue(e.Data);
        };
        Process.ErrorDataReceived += (s, e) => {
            if (e.Data != null) OutputLines.Enqueue(e.Data);
        };
        Process.Exited += (s, e) => {
            HasExited = true;
            try { ExitCode = Process.ExitCode; } catch {}
        };

        bool started = Process.Start();
        if (started) {
            Process.BeginOutputReadLine();
            Process.BeginErrorReadLine();
        }
        return started;
    }

    public void Kill() {
        try {
            if (Process != null && !Process.HasExited) {
                Process.Kill();
            }
        } catch {}
    }
}
'@
}

# ==============================================================================
# Settings Persistence
# ==============================================================================

function Get-GuiConfigPath {
    if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
        $localCfg = Join-Path $PSScriptRoot "quantize-rocmfpx-gui.json"
        if (Test-Path $localCfg) { return $localCfg }
    }
    $appDataDir = Join-Path $env:LOCALAPPDATA "llama-cpp-gui"
    return (Join-Path $appDataDir "settings.json")
}

function Load-GuiConfig {
    $cfg = [PSCustomObject]@{}
    $appDataPath = Join-Path $env:LOCALAPPDATA "llama-cpp-gui\settings.json"
    if (Test-Path $appDataPath) {
        try {
            $raw = Get-Content $appDataPath -Raw -ErrorAction Stop
            if (-not [string]::IsNullOrWhiteSpace($raw)) {
                $cfg = $raw | ConvertFrom-Json
            }
        } catch {}
    }
    if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
        $localPath = Join-Path $PSScriptRoot "quantize-rocmfpx-gui.json"
        if (Test-Path $localPath) {
            try {
                $rawLocal = Get-Content $localPath -Raw -ErrorAction Stop
                if (-not [string]::IsNullOrWhiteSpace($rawLocal)) {
                    $localObj = $rawLocal | ConvertFrom-Json
                    foreach ($prop in $localObj.PSObject.Properties) {
                        $cfg | Add-Member -NotePropertyName $prop.Name -NotePropertyValue $prop.Value -Force
                    }
                }
            } catch {}
        }
    }
    return $cfg
}

function Save-GuiConfig {
    param(
        [string]$QuantizeExe = "",
        [string]$ImatrixExe = ""
    )
    try {
        $cfg = Load-GuiConfig
        if (-not [string]::IsNullOrWhiteSpace($QuantizeExe) -and (Test-Path $QuantizeExe)) {
            $resolvedQuant = (Resolve-Path $QuantizeExe).Path
            $cfg | Add-Member -NotePropertyName "QuantizeExe" -NotePropertyValue $resolvedQuant -Force
        }
        if (-not [string]::IsNullOrWhiteSpace($ImatrixExe) -and (Test-Path $ImatrixExe)) {
            $resolvedImatrix = (Resolve-Path $ImatrixExe).Path
            $cfg | Add-Member -NotePropertyName "ImatrixExe" -NotePropertyValue $resolvedImatrix -Force
        }
        $cfgPath = Get-GuiConfigPath
        $cfgDir = Split-Path -Parent $cfgPath
        if ($cfgDir -and -not (Test-Path $cfgDir)) {
            New-Item -ItemType Directory -Path $cfgDir -Force | Out-Null
        }
        $cfg | ConvertTo-Json -Depth 4 | Set-Content -Path $cfgPath -Force -Encoding utf8
    } catch {}
}

# ==============================================================================
# Business Logic
# ==============================================================================

function Find-QuantizeBinary {
    param([string]$ScriptDir)

    $savedCfg = Load-GuiConfig
    if ($savedCfg.PSObject.Properties['QuantizeExe'] -and $savedCfg.QuantizeExe -and (Test-Path $savedCfg.QuantizeExe)) {
        return [PSCustomObject]@{
            Path = (Resolve-Path $savedCfg.QuantizeExe).Path
            Origin = "Saved Settings"
        }
    }

    $Candidates = @(
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "llama-quantize.exe"); Source = "Script Directory" },
        [PSCustomObject]@{ Path = (Join-Path (Get-Location) "llama-quantize.exe"); Source = "Current Working Directory" },
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "..\llama-quantize.exe"); Source = "Parent Directory" },
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "..\build\bin\llama-quantize.exe"); Source = "Build Directory" },
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "..\build\bin\Release\llama-quantize.exe"); Source = "Build Release Directory" },
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "..\build-rocm\bin\llama-quantize.exe"); Source = "ROCm Build Directory" }
    )

    foreach ($cand in $Candidates) {
        if (Test-Path $cand.Path) {
            return [PSCustomObject]@{
                Path = (Resolve-Path $cand.Path).Path
                Origin = $cand.Source
            }
        }
    }

    $cmd = Get-Command "llama-quantize" -ErrorAction SilentlyContinue
    if ($cmd) {
        return [PSCustomObject]@{
            Path = $cmd.Source
            Origin = "System PATH"
        }
    }

    return $null
}

function Find-ImatrixBinary {
    param(
        [string]$ScriptDir,
        [string]$KnownQuantizeBin = ""
    )

    $savedCfg = Load-GuiConfig
    if ($savedCfg.PSObject.Properties['ImatrixExe'] -and $savedCfg.ImatrixExe -and (Test-Path $savedCfg.ImatrixExe)) {
        return [PSCustomObject]@{
            Path = (Resolve-Path $savedCfg.ImatrixExe).Path
            Origin = "Saved Settings"
        }
    }

    $Candidates = @()
    if ($KnownQuantizeBin -and (Test-Path $KnownQuantizeBin)) {
        $binDir = Split-Path -Parent $KnownQuantizeBin
        $Candidates += [PSCustomObject]@{ Path = (Join-Path $binDir "llama-imatrix.exe"); Source = "Directory of llama-quantize" }
    }
    $Candidates += @(
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "llama-imatrix.exe"); Source = "Script Directory" },
        [PSCustomObject]@{ Path = (Join-Path (Get-Location) "llama-imatrix.exe"); Source = "Current Working Directory" },
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "..\llama-imatrix.exe"); Source = "Parent Directory" },
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "..\build\bin\llama-imatrix.exe"); Source = "Build Directory" },
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "..\build\bin\Release\llama-imatrix.exe"); Source = "Build Release Directory" },
        [PSCustomObject]@{ Path = (Join-Path $ScriptDir "..\build-rocm\bin\llama-imatrix.exe"); Source = "ROCm Build Directory" }
    )

    foreach ($cand in $Candidates) {
        if (Test-Path $cand.Path) {
            return [PSCustomObject]@{
                Path = (Resolve-Path $cand.Path).Path
                Origin = $cand.Source
            }
        }
    }

    $cmd = Get-Command "llama-imatrix" -ErrorAction SilentlyContinue
    if ($cmd) {
        return [PSCustomObject]@{
            Path = $cmd.Source
            Origin = "System PATH"
        }
    }

    return $null
}

function Set-RocmExecutionEnvironment {
    param([int]$GpuLayers = 0)

    if ($GpuLayers -le 0) {
        # Pure CPU execution: isolate ROCm to prevent device initialization crashes
        $env:HIP_VISIBLE_DEVICES = ""
        return "CPU (ROCm bypassed)"
    }

    # Detect multi-GPU setups where Device 0 is an unsupported iGPU (gfx1036) and Device 1 is a dedicated AMD GPU (e.g. RX 9060 XT / gfx1200)
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
        $hasDgpu = $gpus | Where-Object { $_ -match 'Radeon RX' }
        $hasIgpu = $gpus | Where-Object { $_ -match 'Radeon\(TM\) Graphics' -or $_ -match 'Radeon Graphics' }
        if ($hasDgpu -and $hasIgpu) {
            # Dedicated RX 9060 XT is Device 1; iGPU Device 0 causes gfx1036 kernel image invalid aborts
            $env:HIP_VISIBLE_DEVICES = "1"
            return "Device 1 (Dedicated RX 9060 XT isolated)"
        }
    } catch {}

    # If HIP_VISIBLE_DEVICES is already explicitly set to something specific (not default 0,1), preserve it
    if (-not [string]::IsNullOrWhiteSpace($env:HIP_VISIBLE_DEVICES) -and $env:HIP_VISIBLE_DEVICES -ne "0,1") {
        return "Custom ($env:HIP_VISIBLE_DEVICES)"
    }

    $env:HIP_VISIBLE_DEVICES = "0"
    return "Device 0"
}

function Get-SuggestedOutputPath {
    param(
        [string]$SourcePath,
        [string]$Preset
    )

    if ([string]::IsNullOrWhiteSpace($SourcePath)) {
        return ""
    }

    $dir = Split-Path -Parent $SourcePath
    $filename = Split-Path -Leaf $SourcePath
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($filename)

    # Strip split markers if present (e.g., -00001-of-00002)
    $cleanBase = $baseName -replace "-\d{5}-of-\d{5}$", ""
    # Strip existing quant tags if present (e.g., -F16, -BF16, -Q8_0, -Q4_K_M)
    $cleanBase = $cleanBase -replace "-(BF16|F16|Q8_0|Q4_K_M|Q4_0|Q6_K|Q5_K_M|f16|bf16)$", ""
    $cleanBase = $cleanBase -replace "-(Q[0-9]_[0-9A-Z_]+|tq[0-9]_[0-9a-z]+)$", ""

    $newName = "$cleanBase-$Preset.gguf"
    if ($dir) {
        return Join-Path $dir $newName
    }
    return $newName
}

function Get-SuggestedImatrixPath {
    param([string]$SourcePath)

    if ([string]::IsNullOrWhiteSpace($SourcePath)) {
        return ""
    }

    $dir = Split-Path -Parent $SourcePath
    $filename = Split-Path -Leaf $SourcePath
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($filename)

    # Strip split markers if present (e.g., -00001-of-00002)
    $cleanBase = $baseName -replace "-\d{5}-of-\d{5}$", ""
    $cleanBase = $cleanBase -replace "-(BF16|F16|Q8_0|Q4_K_M|Q4_0|Q6_K|Q5_K_M|f16|bf16)$", ""
    $cleanBase = $cleanBase -replace "-(Q[0-9]_[0-9A-Z_]+|tq[0-9]_[0-9a-z]+)$", ""

    $newName = "$cleanBase-imatrix.gguf"
    if ($dir) {
        return Join-Path $dir $newName
    }
    return $newName
}

function Test-GgufSplitsComplete {
    param([string]$FilePath)

    if ([string]::IsNullOrWhiteSpace($FilePath) -or -not (Test-Path $FilePath)) {
        return @{ Complete = $true; Missing = @() }
    }

    $fileName = Split-Path -Leaf $FilePath
    $dir = Split-Path -Parent $FilePath

    # Pattern: Name-00001-of-00002.gguf
    if ($fileName -match '^(.*)-(\d{5})-of-(\d{5})\.gguf$') {
        $prefix = $Matches[1]
        $totalDigits = $Matches[3]
        $totalSplits = [int]$totalDigits
        $missing = @()

        for ($i = 1; $i -le $totalSplits; $i++) {
            $splitIndex = "{0:D5}" -f $i
            $splitName = "$prefix-$splitIndex-of-$totalDigits.gguf"
            $splitPath = if ($dir) { Join-Path $dir $splitName } else { $splitName }
            if (-not (Test-Path $splitPath)) {
                $missing += $splitName
            }
        }

        if ($missing.Count -gt 0) {
            return @{ Complete = $false; Missing = $missing }
        }
    }

    return @{ Complete = $true; Missing = @() }
}

function Build-QuantizeArguments {
    param(
        [string]$Source,
        [string]$Output,
        [string]$Preset,
        [string]$Imatrix,
        [bool]$AllowRequantize,
        [int]$Threads
    )

    $argsList = New-Object System.Collections.Generic.List[string]

    if ($AllowRequantize) {
        $argsList.Add("--allow-requantize")
    }

    if (-not [string]::IsNullOrWhiteSpace($Imatrix)) {
        $argsList.Add("--imatrix")
        $argsList.Add($Imatrix)
    }

    $argsList.Add($Source)
    $argsList.Add($Output)
    $argsList.Add($Preset)

    if ($Threads -gt 0) {
        $argsList.Add($Threads.ToString())
    }

    return $argsList.ToArray()
}

# ==============================================================================
# UI View (XAML)
# ==============================================================================

[xml]$xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="ROCmFPX &amp; TurboQuant Model Quantizer"
        Width="780" Height="730" MinWidth="680" MinHeight="620"
        WindowStartupLocation="CenterScreen"
        Background="#181820" Foreground="#F0F0F5"
        FontFamily="Segoe UI" FontSize="13">

    <Window.Resources>
        <!-- Fix ComboBox System Brushes for Dropdown Popup Contrast -->
        <SolidColorBrush x:Key="{x:Static SystemColors.WindowBrushKey}" Color="#20202A"/>
        <SolidColorBrush x:Key="{x:Static SystemColors.WindowTextBrushKey}" Color="#F0F0F5"/>
        <SolidColorBrush x:Key="{x:Static SystemColors.HighlightBrushKey}" Color="#2563EB"/>
        <SolidColorBrush x:Key="{x:Static SystemColors.HighlightTextBrushKey}" Color="#FFFFFF"/>

        <Style TargetType="ToolTip">
            <Setter Property="Background" Value="#1E1E28"/>
            <Setter Property="Foreground" Value="#F8FAFC"/>
            <Setter Property="BorderBrush" Value="#3B82F6"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="Padding" Value="8,6"/>
            <Setter Property="FontSize" Value="12"/>
        </Style>
        <Style TargetType="TextBlock">
            <Setter Property="Foreground" Value="#E0E0EB"/>
        </Style>
        <Style TargetType="TextBox">
            <Setter Property="Background" Value="#22222C"/>
            <Setter Property="Foreground" Value="#FFFFFF"/>
            <Setter Property="BorderBrush" Value="#383848"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="Padding" Value="6,4"/>
            <Setter Property="VerticalContentAlignment" Value="Center"/>
            <Style.Triggers>
                <Trigger Property="IsEnabled" Value="False">
                    <Setter Property="Background" Value="#1A1A22"/>
                    <Setter Property="Foreground" Value="#64748B"/>
                    <Setter Property="BorderBrush" Value="#2D2D3B"/>
                </Trigger>
            </Style.Triggers>
        </Style>
        <Style TargetType="{x:Type Button}">
            <Setter Property="Background" Value="#2E2E3D"/>
            <Setter Property="Foreground" Value="#FFFFFF"/>
            <Setter Property="BorderBrush" Value="#45455A"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="Padding" Value="12,5"/>
            <Setter Property="Cursor" Value="Hand"/>
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="{x:Type Button}">
                        <Border x:Name="border"
                                Background="{TemplateBinding Background}"
                                BorderBrush="{TemplateBinding BorderBrush}"
                                BorderThickness="{TemplateBinding BorderThickness}"
                                CornerRadius="4"
                                SnapsToDevicePixels="True">
                            <ContentPresenter x:Name="contentPresenter"
                                              Focusable="False"
                                              HorizontalAlignment="Center"
                                              VerticalAlignment="Center"
                                              Margin="{TemplateBinding Padding}"
                                              RecognizesAccessKey="True"
                                              SnapsToDevicePixels="{TemplateBinding SnapsToDevicePixels}"/>
                        </Border>
                        <ControlTemplate.Triggers>
                            <Trigger Property="IsMouseOver" Value="True">
                                <Setter Property="Opacity" Value="0.9"/>
                            </Trigger>
                            <Trigger Property="IsPressed" Value="True">
                                <Setter Property="Opacity" Value="0.75"/>
                            </Trigger>
                            <Trigger Property="IsEnabled" Value="False">
                                <Setter TargetName="border" Property="Background" Value="#20202A"/>
                                <Setter TargetName="border" Property="BorderBrush" Value="#2D2D3B"/>
                                <Setter Property="Foreground" Value="#64748B"/>
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>
        <ControlTemplate x:Key="ComboBoxToggleButtonTemplate" TargetType="{x:Type ToggleButton}">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition />
                    <ColumnDefinition Width="26" />
                </Grid.ColumnDefinitions>
                <Border x:Name="border"
                        Grid.ColumnSpan="2"
                        Background="{TemplateBinding Background}"
                        BorderBrush="{TemplateBinding BorderBrush}"
                        BorderThickness="{TemplateBinding BorderThickness}"
                        CornerRadius="4"
                        SnapsToDevicePixels="True" />
                <Path x:Name="arrow"
                      Grid.Column="1"
                      HorizontalAlignment="Center"
                      VerticalAlignment="Center"
                      Data="M 0 0 L 4 4 L 8 0 Z"
                      Fill="#94A3B8" />
            </Grid>
            <ControlTemplate.Triggers>
                <Trigger Property="IsMouseOver" Value="True">
                    <Setter TargetName="border" Property="BorderBrush" Value="#60A5FA" />
                    <Setter TargetName="arrow" Property="Fill" Value="#F8FAFC" />
                </Trigger>
                <Trigger Property="IsChecked" Value="True">
                    <Setter TargetName="border" Property="BorderBrush" Value="#3B82F6" />
                    <Setter TargetName="arrow" Property="Fill" Value="#3B82F6" />
                </Trigger>
                <Trigger Property="IsEnabled" Value="False">
                    <Setter TargetName="border" Property="Background" Value="#1A1A22" />
                    <Setter TargetName="border" Property="BorderBrush" Value="#2D2D3B" />
                    <Setter TargetName="arrow" Property="Fill" Value="#4B5563" />
                </Trigger>
            </ControlTemplate.Triggers>
        </ControlTemplate>
        <Style TargetType="{x:Type ComboBox}">
            <Setter Property="Background" Value="#22222C" />
            <Setter Property="Foreground" Value="#F8FAFC" />
            <Setter Property="BorderBrush" Value="#383848" />
            <Setter Property="BorderThickness" Value="1" />
            <Setter Property="Padding" Value="8,4" />
            <Setter Property="VerticalContentAlignment" Value="Center" />
            <Setter Property="SnapsToDevicePixels" Value="True" />
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="{x:Type ComboBox}">
                        <Grid>
                            <ToggleButton x:Name="ToggleButton"
                                          Focusable="False"
                                          IsChecked="{Binding Path=IsDropDownOpen, Mode=TwoWay, RelativeSource={RelativeSource TemplatedParent}}"
                                          ClickMode="Press"
                                          Background="{TemplateBinding Background}"
                                          BorderBrush="{TemplateBinding BorderBrush}"
                                          BorderThickness="{TemplateBinding BorderThickness}"
                                          Template="{StaticResource ComboBoxToggleButtonTemplate}" />
                            <ContentPresenter x:Name="ContentSite"
                                              IsHitTestVisible="False"
                                              Content="{TemplateBinding SelectionBoxItem}"
                                              ContentTemplate="{TemplateBinding SelectionBoxItemTemplate}"
                                              ContentTemplateSelector="{TemplateBinding ItemTemplateSelector}"
                                              Margin="8,4,28,4"
                                              VerticalAlignment="Center"
                                              HorizontalAlignment="Left">
                                <ContentPresenter.Resources>
                                    <Style TargetType="{x:Type TextBlock}">
                                        <Setter Property="Foreground" Value="#F8FAFC" />
                                    </Style>
                                </ContentPresenter.Resources>
                            </ContentPresenter>
                            <Popup x:Name="Popup"
                                   Placement="Bottom"
                                   IsOpen="{TemplateBinding IsDropDownOpen}"
                                   AllowsTransparency="True"
                                   Focusable="False"
                                   PopupAnimation="Slide">
                                <Grid x:Name="DropDown"
                                      SnapsToDevicePixels="True"
                                      MinWidth="{TemplateBinding ActualWidth}"
                                      MaxHeight="{TemplateBinding MaxDropDownHeight}">
                                    <Border x:Name="DropDownBorder"
                                            Background="#1E1E28"
                                            BorderBrush="#3B82F6"
                                            BorderThickness="1"
                                            CornerRadius="4"
                                            Margin="0,2,0,2">
                                        <ScrollViewer SnapsToDevicePixels="True">
                                            <StackPanel IsItemsHost="True" KeyboardNavigation.DirectionalNavigation="Cycle" />
                                        </ScrollViewer>
                                    </Border>
                                </Grid>
                            </Popup>
                        </Grid>
                        <ControlTemplate.Triggers>
                            <Trigger Property="HasItems" Value="False">
                                <Setter TargetName="DropDownBorder" Property="MinHeight" Value="95" />
                            </Trigger>
                            <Trigger Property="IsMouseOver" Value="True">
                                <Setter Property="BorderBrush" Value="#60A5FA" />
                            </Trigger>
                            <Trigger Property="IsEnabled" Value="False">
                                <Setter Property="Foreground" Value="#64748B" />
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>
        <Style TargetType="{x:Type ComboBoxItem}">
            <Setter Property="Background" Value="#1E1E28" />
            <Setter Property="Foreground" Value="#F8FAFC" />
            <Setter Property="Padding" Value="8,6" />
            <Setter Property="BorderThickness" Value="0" />
            <Setter Property="SnapsToDevicePixels" Value="True" />
            <Style.Triggers>
                <Trigger Property="IsHighlighted" Value="True">
                    <Setter Property="Background" Value="#2563EB" />
                    <Setter Property="Foreground" Value="#FFFFFF" />
                </Trigger>
                <Trigger Property="IsSelected" Value="True">
                    <Setter Property="Background" Value="#1D4ED8" />
                    <Setter Property="Foreground" Value="#FFFFFF" />
                </Trigger>
            </Style.Triggers>
        </Style>
        <Style TargetType="CheckBox">
            <Setter Property="Foreground" Value="#E0E0EB"/>
            <Setter Property="VerticalContentAlignment" Value="Center"/>
        </Style>
    </Window.Resources>

    <Grid Margin="18">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/> <!-- Title Header -->
            <RowDefinition Height="Auto"/> <!-- Executable Section -->
            <RowDefinition Height="Auto"/> <!-- Files Section -->
            <RowDefinition Height="Auto"/> <!-- Options Section -->
            <RowDefinition Height="Auto"/> <!-- Actions Section -->
            <RowDefinition Height="*"/>    <!-- Log Output -->
        </Grid.RowDefinitions>

        <!-- 1. Header -->
        <Border Grid.Row="0" Margin="0,0,0,14" Padding="0,0,0,10" BorderBrush="#2F2F3D" BorderThickness="0,0,0,1">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>
                <StackPanel Grid.Column="0">
                    <TextBlock Text="ROCmFPX &amp; TurboQuant Quantizer" FontSize="20" FontWeight="SemiBold" Foreground="#60A5FA"/>
                    <TextBlock Text="AMD ROCm / RDNA tensor acceleration &amp; WHT-rotated cache/weight quantization" FontSize="11" Foreground="#94A3B8" Margin="0,2,0,0"/>
                </StackPanel>
                <TextBlock Grid.Column="1" Text="v1.0 (WPF)" VerticalAlignment="Center" FontSize="11" Foreground="#64748B"/>
            </Grid>
        </Border>

        <!-- 2. Executable Section -->
        <Border Grid.Row="1" Background="#20202A" CornerRadius="6" Padding="12" Margin="0,0,0,12" BorderBrush="#2D2D3B" BorderThickness="1">
            <Grid>
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                </Grid.RowDefinitions>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>

                <TextBlock Grid.Row="0" Grid.ColumnSpan="2" Text="Quantization Executable (llama-quantize.exe)" FontWeight="SemiBold" Margin="0,0,0,6" ToolTip="Path to the llama-quantize binary that performs weight quantization"/>
                
                <TextBox Grid.Row="1" Grid.Column="0" Name="TxtExePath" Height="30" Margin="0,0,8,0" ToolTip="Full path to llama-quantize.exe binary"/>
                <Button Grid.Row="1" Grid.Column="1" Name="BtnBrowseExe" Content="Browse..." Width="100" Height="30" HorizontalAlignment="Right" ToolTip="Browse filesystem for llama-quantize.exe"/>

                <TextBlock Grid.Row="2" Grid.ColumnSpan="2" Name="LblExeStatus" Text="Searching for llama-quantize.exe..." FontSize="11" Foreground="#10B981" Margin="2,5,0,0"/>
            </Grid>
        </Border>

        <!-- 3. Model Files & Preset Section -->
        <Border Grid.Row="2" Background="#20202A" CornerRadius="6" Padding="12" Margin="0,0,0,12" BorderBrush="#2D2D3B" BorderThickness="1">
            <Grid>
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto"/> <!-- Source Model -->
                    <RowDefinition Height="Auto"/> <!-- Preset -->
                    <RowDefinition Height="Auto"/> <!-- Output Model -->
                    <RowDefinition Height="Auto"/> <!-- Imatrix -->
                </Grid.RowDefinitions>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="130"/>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>

                <!-- Source Model -->
                <TextBlock Grid.Row="0" Grid.Column="0" Text="Source Model:" VerticalAlignment="Center" Margin="0,0,0,10" ToolTip="Path to input GGUF model (F16/BF16 or existing quantized model)"/>
                <TextBox Grid.Row="0" Grid.Column="1" Name="TxtSourceModel" Height="30" Margin="0,0,8,10" ToolTip="Input GGUF model path (typically unquantized F16/BF16, or Q8_0/Q4_K_M if requantizing)"/>
                <Button Grid.Row="0" Grid.Column="2" Name="BtnBrowseSource" Content="Browse..." Width="100" Height="30" Margin="0,0,0,10" HorizontalAlignment="Right" ToolTip="Browse filesystem for input GGUF model file"/>

                <!-- Preset Selection -->
                <TextBlock Grid.Row="1" Grid.Column="0" Text="Quant Preset:" VerticalAlignment="Center" Margin="0,0,0,10" ToolTip="Target quantization format (ROCmFP4, ROCmFPX, TurboQuant, or upstream standard)"/>
                <ComboBox Grid.Row="1" Grid.Column="1" Name="CmbPreset" Height="30" Margin="0,0,8,10" ToolTip="Select quantization preset format">
                    <!-- ROCmFP4 -->
                    <ComboBoxItem Content="Q4_0_ROCMFP4_FAST - Fastest 4-Bit FP4 (Recommended)" Tag="Q4_0_ROCMFP4_FAST" IsSelected="True" ToolTip="4.25 bpw. Fastest 4-bit floating-point format for AMD RDNA3 and RDNA4 GPUs. Native single-scale speed layout."/>
                    <ComboBoxItem Content="Q4_0_ROCMFP4 - Standard 4-Bit FP4" Tag="Q4_0_ROCMFP4" ToolTip="4.50 bpw. Standard ROCm FP4 format with UE4M3 dual scales. Balanced performance and quality."/>
                    <ComboBoxItem Content="Q4_0_ROCMFP4_COHERENT - Coherent 4-Bit FP4 (Higher Quality)" Tag="Q4_0_ROCMFP4_COHERENT" ToolTip="4.70 bpw. Coherent 4-bit FP4 with Q6_K token embeddings for improved response coherence."/>
                    <ComboBoxItem Content="Q4_0_ROCMFP4_STRIX - Tuned for Strix Point iGPU" Tag="Q4_0_ROCMFP4_STRIX" ToolTip="~4.49 bpw. Optimized for AMD Strix Point / Strix Halo APUs (Zen 5 + RDNA 3.5 iGPU) with high-quality attention K/V recipe."/>
                    <ComboBoxItem Content="Q4_0_ROCMFP4_STRIX_LEAN - Lean Strix Point Format" Tag="Q4_0_ROCMFP4_STRIX_LEAN" ToolTip="~4.38 bpw. Lean variant for Strix Point iGPUs with Q5_K token embeddings to conserve memory."/>
                    <!-- ROCmFP3 -->
                    <ComboBoxItem Content="Q3_0_ROCMFPX - 3-Bit FP3 (Fastest 3-Bit)" Tag="Q3_0_ROCMFPX" ToolTip="3.50 bpw. 3-bit floating-point quantization for AMD GPUs with fast ROCm/Vulkan staging. Maximum VRAM savings."/>
                    <ComboBoxItem Content="Q3_0_ROCMFPX_AGENT - 3-Bit FP3 Agent (Best with Imatrix)" Tag="Q3_0_ROCMFPX_AGENT" ToolTip="Agent/tool-call coherent 3-bit ROCmFPx routing. Best quality when paired with an importance matrix (imatrix)."/>
                    <!-- ROCmFP6 -->
                    <ComboBoxItem Content="Q6_0_ROCMFPX - 6-Bit FP6 (High Precision)" Tag="Q6_0_ROCMFPX" ToolTip="6.50 bpw. 6-bit floating-point format for high-precision inference with accelerated ROCm kernels."/>
                    <ComboBoxItem Content="Q6_0_ROCMFPX_AGENT - 6-Bit FP6 Agent (Near Lossless)" Tag="Q6_0_ROCMFPX_AGENT" ToolTip="Agent/tool-call coherent 6-bit ROCmFPx routing. Near-lossless precision for complex reasoning."/>
                    <ComboBoxItem Content="Q6_0_ROCMFPX_LEAN - 6-Bit FP6 Lean" Tag="Q6_0_ROCMFPX_LEAN" ToolTip="Size/speed-biased 6-bit ROCmFPx routing without heavy Q8 layer boosts."/>
                    <!-- ROCmFP8 -->
                    <ComboBoxItem Content="Q8_0_ROCMFPX - 8-Bit FP8" Tag="Q8_0_ROCMFPX" ToolTip="8.25 bpw. 8-bit floating-point format with ROCm tensor acceleration. Reference-grade accuracy."/>
                    <ComboBoxItem Content="Q8_0_ROCMFPX_AGENT - 8-Bit FP8 Agent" Tag="Q8_0_ROCMFPX_AGENT" ToolTip="Agent-guided 8-bit ROCmFPx format with critical tensor protection."/>
                    <!-- ROCm Integer -->
                    <ComboBoxItem Content="Q4_0_ROCMI4 - 4-Bit Integer ROCm" Tag="Q4_0_ROCMI4" ToolTip="4.25 bpw. Native signed-nibble 4-bit integer format without codebook for ROCm matrix cores."/>
                    <!-- TurboQuant Weights -->
                    <ComboBoxItem Content="tq3_1s - TurboQuant 3-Bit (WHT-rotated Lloyd-Max)" Tag="tq3_1s" ToolTip="4.00 bpw. TurboQuant 3-bit weight format with Walsh-Hadamard rotation (WHT) and Lloyd-Max codebooks."/>
                    <ComboBoxItem Content="tq4_1s - TurboQuant 4-Bit (WHT-rotated Lloyd-Max)" Tag="tq4_1s" ToolTip="5.00 bpw. TurboQuant 4-bit weight format with Walsh-Hadamard rotation (WHT) and Lloyd-Max codebooks."/>
                    <!-- Upstream Standard Formats -->
                    <ComboBoxItem Content="Q4_K_M - Upstream 4-Bit K-Quant" Tag="Q4_K_M" ToolTip="4.58 bpw. Upstream medium 4-bit K-quant with mixed tensor precision. Popular general-purpose format."/>
                    <ComboBoxItem Content="Q5_K_M - Upstream 5-Bit K-Quant" Tag="Q5_K_M" ToolTip="5.33 bpw. Upstream medium 5-bit K-quant. High accuracy with moderate VRAM usage."/>
                    <ComboBoxItem Content="Q6_K - Upstream 6-Bit K-Quant" Tag="Q6_K" ToolTip="6.14 bpw. Upstream 6-bit K-quant. Very close to F16 quality."/>
                    <ComboBoxItem Content="Q8_0 - Upstream 8-Bit Standard" Tag="Q8_0" ToolTip="7.96 bpw. Upstream standard 8-bit quantization. Highest accuracy among standard integer quants."/>
                    <ComboBoxItem Content="IQ3_S - Upstream 3-Bit I-Quant (Needs Imatrix)" Tag="IQ3_S" ToolTip="3.44 bpw. Upstream 3-bit importance matrix quant. Requires an imatrix file for acceptable quality."/>
                </ComboBox>
                <Button Grid.Row="1" Grid.Column="2" Name="BtnSuggestOutput" Content="Auto-Name" Width="100" Height="30" Margin="0,0,0,10" HorizontalAlignment="Right" ToolTip="Regenerate output path based on source and preset"/>

                <!-- Output Model -->
                <TextBlock Grid.Row="2" Grid.Column="0" Text="Output Model:" VerticalAlignment="Center" Margin="0,0,0,10" ToolTip="Destination path for the quantized GGUF file"/>
                <TextBox Grid.Row="2" Grid.Column="1" Name="TxtOutputModel" Height="30" Margin="0,0,8,10" ToolTip="Path where the quantized GGUF model will be saved"/>
                <Button Grid.Row="2" Grid.Column="2" Name="BtnBrowseOutput" Content="Save As..." Width="100" Height="30" Margin="0,0,0,10" HorizontalAlignment="Right" ToolTip="Choose destination path and filename for output GGUF"/>

                <!-- Importance Matrix -->
                <TextBlock Grid.Row="3" Grid.Column="0" Text="Imatrix (Optional):" VerticalAlignment="Center" ToolTip="Importance matrix file to preserve accuracy in low-bit quants"/>
                <TextBox Grid.Row="3" Grid.Column="1" Name="TxtImatrix" Height="30" Margin="0,0,8,0" ToolTip="Optional importance matrix (.gguf or .dat). Greatly improves 3-bit and 4-bit quantization quality."/>
                <StackPanel Grid.Row="3" Grid.Column="2" Orientation="Horizontal" HorizontalAlignment="Right">
                    <Button Name="BtnBrowseImatrix" Content="Browse..." Width="85" Height="30" Margin="0,0,6,0" ToolTip="Select an existing importance matrix file"/>
                    <Button Name="BtnCreateImatrix" Content="Generate..." Width="110" Height="30" Background="#0D9488" BorderBrush="#14B8A6" ToolTip="Calculate importance matrix from calibration text dataset using GPU-accelerated llama-imatrix"/>
                </StackPanel>
            </Grid>
        </Border>

        <!-- 4. Options Section -->
        <Border Grid.Row="3" Background="#20202A" CornerRadius="6" Padding="12" Margin="0,0,0,12" BorderBrush="#2D2D3B" BorderThickness="1">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                    <ColumnDefinition Width="105"/>
                </Grid.ColumnDefinitions>

                <CheckBox Grid.Column="0" Name="ChkAllowRequantize" Content="Allow Requantize (--allow-requantize)" ToolTip="Check if source is already quantized (e.g. Q8_0 or Q4_K_M) rather than F16/BF16. Requantization runs on CPU."/>

                <TextBlock Grid.Column="1" Text="CPU Threads:" VerticalAlignment="Center" Margin="0,0,8,0" ToolTip="Number of CPU worker threads (0 = auto-detect based on CPU cores)"/>
                <ComboBox Grid.Column="2" Name="CmbThreads" Height="28" ToolTip="CPU threads to use for quantization (offline quantization runs on CPU)">
                    <ComboBoxItem Content="0 (Auto)" Tag="0" IsSelected="True" ToolTip="Automatically detect and use all available CPU logical cores"/>
                    <ComboBoxItem Content="4" Tag="4" ToolTip="Use 4 CPU threads"/>
                    <ComboBoxItem Content="8" Tag="8" ToolTip="Use 8 CPU threads"/>
                    <ComboBoxItem Content="12" Tag="12" ToolTip="Use 12 CPU threads"/>
                    <ComboBoxItem Content="16" Tag="16" ToolTip="Use 16 CPU threads"/>
                    <ComboBoxItem Content="24" Tag="24" ToolTip="Use 24 CPU threads"/>
                    <ComboBoxItem Content="32" Tag="32" ToolTip="Use 32 CPU threads"/>
                </ComboBox>
            </Grid>
        </Border>

        <!-- 5. Actions & Status -->
        <Grid Grid.Row="4" Margin="0,0,0,10">
            <Grid.RowDefinitions>
                <RowDefinition Height="Auto"/>
                <RowDefinition Height="Auto"/>
            </Grid.RowDefinitions>
            <Grid.ColumnDefinitions>
                <ColumnDefinition Width="*"/>
                <ColumnDefinition Width="Auto"/>
                <ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>

            <TextBlock Grid.Row="0" Grid.Column="0" Name="LblStatus" Text="Ready" VerticalAlignment="Center" FontWeight="SemiBold" Foreground="#38BDF8" ToolTip="Current execution status"/>
            <Button Grid.Row="0" Grid.Column="1" Name="BtnCancel" Content="Cancel" Width="100" Height="34" Margin="0,0,10,0" IsEnabled="False" Background="#7F1D1D" BorderBrush="#991B1B" ToolTip="Abort the active quantization process"/>
            <Button Grid.Row="0" Grid.Column="2" Name="BtnStart" Content="Start Quantization" Width="160" Height="34" FontWeight="SemiBold" Background="#2563EB" BorderBrush="#3B82F6" ToolTip="Begin quantization with selected parameters"/>

            <!-- Progress Bar & Indicator -->
            <Grid Grid.Row="1" Grid.ColumnSpan="3" Margin="0,10,0,0">
                <ProgressBar Name="PrgBar" Height="18" Minimum="0" Maximum="100" Value="0" Background="#1E1E28" Foreground="#3B82F6" BorderBrush="#383848" BorderThickness="1"/>
                <TextBlock Name="LblProgressText" Text="Ready" HorizontalAlignment="Center" VerticalAlignment="Center" FontSize="11" Foreground="#F0F0F5" FontWeight="SemiBold"/>
            </Grid>
        </Grid>

        <!-- 6. Log Output -->
        <Border Grid.Row="5" Background="#14141B" CornerRadius="6" BorderBrush="#2D2D3B" BorderThickness="1" Padding="8">
            <TextBox Name="TxtLog" Background="Transparent" Foreground="#E2E8F0" BorderThickness="0"
                     FontFamily="Consolas, Courier New, monospace" FontSize="11"
                     IsReadOnly="True" VerticalScrollBarVisibility="Auto" HorizontalScrollBarVisibility="Auto"
                     AcceptsReturn="True" TextWrapping="NoWrap"/>
        </Border>
    </Grid>
</Window>
"@

# Parse XAML
$reader = New-Object System.Xml.XmlNodeReader $xaml
$window = [System.Windows.Markup.XamlReader]::Load($reader)
$window.Add_SourceInitialized({
    try {
        $helper = New-Object System.Windows.Interop.WindowInteropHelper($window)
        $src = [System.Windows.Interop.HwndSource]::FromHwnd($helper.Handle)
        if ($src -and $src.CompositionTarget) {
            $src.CompositionTarget.RenderMode = [System.Windows.Interop.RenderMode]::SoftwareOnly
        }
    } catch {}
})

# Map controls
$txtExePath           = $window.FindName("TxtExePath")
$btnBrowseExe         = $window.FindName("BtnBrowseExe")
$lblExeStatus         = $window.FindName("LblExeStatus")
$txtSourceModel       = $window.FindName("TxtSourceModel")
$btnBrowseSource      = $window.FindName("BtnBrowseSource")
$cmbPreset            = $window.FindName("CmbPreset")
$btnSuggestOutput     = $window.FindName("BtnSuggestOutput")
$txtOutputModel       = $window.FindName("TxtOutputModel")
$btnBrowseOutput      = $window.FindName("BtnBrowseOutput")
$txtImatrix           = $window.FindName("TxtImatrix")
$btnBrowseImatrix     = $window.FindName("BtnBrowseImatrix")
$btnCreateImatrix     = $window.FindName("BtnCreateImatrix")
$chkAllowRequantize   = $window.FindName("ChkAllowRequantize")
$cmbThreads           = $window.FindName("CmbThreads")
$lblStatus            = $window.FindName("LblStatus")
$btnStart             = $window.FindName("BtnStart")
$btnCancel            = $window.FindName("BtnCancel")
$txtLog               = $window.FindName("TxtLog")
$prgBar               = $window.FindName("PrgBar")
$lblProgressText      = $window.FindName("LblProgressText")

# Store running process and timer references
$script:RunningRunner = $null
$script:QuantTimer = $null
$script:QuantOutputFile = $null

# ==============================================================================
# Controller & Event Wiring
# ==============================================================================

# 1. Initialize Executable Detection
$guiConfig = Load-GuiConfig
$savedQuant = if ($guiConfig.PSObject.Properties['QuantizeExe']) { $guiConfig.QuantizeExe } else { "" }

if ($savedQuant -and (Test-Path $savedQuant)) {
    $txtExePath.Text = $savedQuant
    $lblExeStatus.Text = "Saved executable: $savedQuant"
    $lblExeStatus.Foreground = [System.Windows.Media.Brushes]::LightGreen
} else {
    $foundExe = Find-QuantizeBinary -ScriptDir $PSScriptRoot
    if ($foundExe) {
        $txtExePath.Text = $foundExe.Path
        $lblExeStatus.Text = "Auto-detected ($($foundExe.Origin)): $($foundExe.Path)"
        $lblExeStatus.Foreground = [System.Windows.Media.Brushes]::LightGreen
        Save-GuiConfig -QuantizeExe $foundExe.Path
    } else {
        $lblExeStatus.Text = "llama-quantize.exe not found automatically. Please browse and select it."
        $lblExeStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
    }
}

# Apply initial values if provided
if ($InitialSource) {
    $txtSourceModel.Text = $InitialSource
    $selectedPresetTag = $cmbPreset.SelectedItem.Tag
    $txtOutputModel.Text = Get-SuggestedOutputPath -SourcePath $InitialSource -Preset $selectedPresetTag
}
if ($InitialImatrix) {
    $txtImatrix.Text = $InitialImatrix
}

# Helper to get current preset tag
function Get-SelectedPresetTag {
    if ($cmbPreset.SelectedItem -and $cmbPreset.SelectedItem.Tag) {
        return $cmbPreset.SelectedItem.Tag.ToString()
    }
    return "Q4_0_ROCMFP4_FAST"
}

# Helper to get selected threads
function Get-SelectedThreads {
    if ($cmbThreads.SelectedItem -and $cmbThreads.SelectedItem.Tag) {
        return [int]$cmbThreads.SelectedItem.Tag
    }
    return 0
}

# Event: Browse Executable
$btnBrowseExe.Add_Click({
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = "Select llama-quantize Executable"
    $dlg.Filter = "Executable files (*.exe)|*.exe|All files (*.*)|*.*"
    if (Test-Path $txtExePath.Text) {
        $dlg.InitialDirectory = Split-Path -Parent $txtExePath.Text
    } elseif (Test-Path $PSScriptRoot) {
        $dlg.InitialDirectory = $PSScriptRoot
    }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $txtExePath.Text = $dlg.FileName
        $lblExeStatus.Text = "Saved executable: $($dlg.FileName)"
        $lblExeStatus.Foreground = [System.Windows.Media.Brushes]::LightGreen
        Save-GuiConfig -QuantizeExe $dlg.FileName
    }
})

# Event: Executable Path Lost Focus (Manual Input)
$txtExePath.Add_LostFocus({
    $trimmed = $txtExePath.Text.Trim()
    if ($trimmed -and (Test-Path $trimmed)) {
        Save-GuiConfig -QuantizeExe $trimmed
        $lblExeStatus.Text = "Saved executable: $trimmed"
        $lblExeStatus.Foreground = [System.Windows.Media.Brushes]::LightGreen
    }
})

# Event: Browse Source Model
$btnBrowseSource.Add_Click({
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = "Select Source GGUF Model"
    $dlg.Filter = "GGUF Model files (*.gguf)|*.gguf|All files (*.*)|*.*"
    if (Test-Path $txtSourceModel.Text) {
        $dlg.InitialDirectory = Split-Path -Parent $txtSourceModel.Text
    }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $txtSourceModel.Text = $dlg.FileName
        $txtOutputModel.Text = Get-SuggestedOutputPath -SourcePath $dlg.FileName -Preset (Get-SelectedPresetTag)
        $splitCheck = Test-GgufSplitsComplete -FilePath $dlg.FileName
        if (-not $splitCheck.Complete) {
            $missingNames = $splitCheck.Missing -join "`r`n  - "
            [System.Windows.MessageBox]::Show("Achtung: Dies ist ein mehrteiliges GGUF-Modell, aber folgende Split-Dateien fehlen im Ordner:`r`n  - $missingNames`r`n`r`nBitte laden Sie alle Teile vor Beginn der Quantisierung herunter.", "Unvollstaendiges Modell", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning)
        }
    }
})

# Event: Preset Selection Changed
$cmbPreset.Add_SelectionChanged({
    if (-not [string]::IsNullOrWhiteSpace($txtSourceModel.Text)) {
        $txtOutputModel.Text = Get-SuggestedOutputPath -SourcePath $txtSourceModel.Text -Preset (Get-SelectedPresetTag)
    }
})

# Event: Suggest Output Button
$btnSuggestOutput.Add_Click({
    if (-not [string]::IsNullOrWhiteSpace($txtSourceModel.Text)) {
        $txtOutputModel.Text = Get-SuggestedOutputPath -SourcePath $txtSourceModel.Text -Preset (Get-SelectedPresetTag)
    }
})

# Event: Browse Output Model
$btnBrowseOutput.Add_Click({
    $dlg = New-Object System.Windows.Forms.SaveFileDialog
    $dlg.Title = "Save Quantized GGUF Model As"
    $dlg.Filter = "GGUF Model files (*.gguf)|*.gguf|All files (*.*)|*.*"
    $dlg.DefaultExt = "gguf"
    if ($txtOutputModel.Text) {
        $dlg.FileName = Split-Path -Leaf $txtOutputModel.Text
        $parent = Split-Path -Parent $txtOutputModel.Text
        if ($parent -and (Test-Path $parent)) {
            $dlg.InitialDirectory = $parent
        }
    }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $txtOutputModel.Text = $dlg.FileName
    }
})

# Event: Browse Imatrix
$btnBrowseImatrix.Add_Click({
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = "Select Importance Matrix (imatrix)"
    $dlg.Filter = "GGUF / Data files (*.gguf;*.dat)|*.gguf;*.dat|All files (*.*)|*.*"
    if ($txtImatrix.Text -and (Test-Path $txtImatrix.Text)) {
        $dlg.InitialDirectory = Split-Path -Parent $txtImatrix.Text
    }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $txtImatrix.Text = $dlg.FileName
    }
})

# Function to launch interactive Imatrix Generator Modal
function Show-ImatrixDialog {
    param(
        [System.Windows.Window]$OwnerWindow,
        [string]$CurrentSource,
        [string]$CurrentBin,
        [string]$CurrentCalibration = ""
    )

    $guiConfig = Load-GuiConfig
    $savedImatrix = if ($guiConfig.PSObject.Properties['ImatrixExe']) { $guiConfig.ImatrixExe } else { "" }

    $defaultImatrixBin = ""
    if ($savedImatrix -and (Test-Path $savedImatrix)) {
        $defaultImatrixBin = $savedImatrix
    } else {
        $imatrixBinObj = Find-ImatrixBinary -ScriptDir $PSScriptRoot -KnownQuantizeBin $CurrentBin
        if ($imatrixBinObj) {
            $defaultImatrixBin = $imatrixBinObj.Path
            Save-GuiConfig -ImatrixExe $defaultImatrixBin
        }
    }
    $defaultImatrixOut = Get-SuggestedImatrixPath -SourcePath $CurrentSource

    $dialogXaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="Generate Importance Matrix (llama-imatrix)"
        Width="740" Height="640" WindowStartupLocation="CenterOwner"
        Background="#191921" Foreground="#E2E8F0" FontFamily="Segoe UI"
        ResizeMode="CanResizeWithGrip">
    <Window.Resources>
        <SolidColorBrush x:Key="{x:Static SystemColors.WindowBrushKey}" Color="#20202A"/>
        <SolidColorBrush x:Key="{x:Static SystemColors.WindowTextBrushKey}" Color="#F0F0F5"/>
        <SolidColorBrush x:Key="{x:Static SystemColors.HighlightBrushKey}" Color="#0D9488"/>
        <SolidColorBrush x:Key="{x:Static SystemColors.HighlightTextBrushKey}" Color="#FFFFFF"/>
        <Style TargetType="ToolTip">
            <Setter Property="Background" Value="#1E1E28"/>
            <Setter Property="Foreground" Value="#F8FAFC"/>
            <Setter Property="BorderBrush" Value="#14B8A6"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="Padding" Value="8,6"/>
            <Setter Property="FontSize" Value="12"/>
        </Style>
        <Style TargetType="TextBox">
            <Setter Property="Background" Value="#2A2A38"/>
            <Setter Property="Foreground" Value="#F8FAFC"/>
            <Setter Property="BorderBrush" Value="#3F3F52"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="Padding" Value="6,4"/>
            <Setter Property="VerticalContentAlignment" Value="Center"/>
            <Style.Triggers>
                <Trigger Property="IsEnabled" Value="False">
                    <Setter Property="Background" Value="#1A1A22"/>
                    <Setter Property="Foreground" Value="#64748B"/>
                    <Setter Property="BorderBrush" Value="#2D2D3B"/>
                </Trigger>
            </Style.Triggers>
        </Style>
        <Style TargetType="{x:Type Button}">
            <Setter Property="Background" Value="#3B82F6"/>
            <Setter Property="Foreground" Value="#FFFFFF"/>
            <Setter Property="BorderBrush" Value="#60A5FA"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="Cursor" Value="Hand"/>
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="{x:Type Button}">
                        <Border x:Name="border"
                                Background="{TemplateBinding Background}"
                                BorderBrush="{TemplateBinding BorderBrush}"
                                BorderThickness="{TemplateBinding BorderThickness}"
                                CornerRadius="4"
                                SnapsToDevicePixels="True">
                            <ContentPresenter x:Name="contentPresenter"
                                              Focusable="False"
                                              HorizontalAlignment="Center"
                                              VerticalAlignment="Center"
                                              Margin="{TemplateBinding Padding}"
                                              RecognizesAccessKey="True"
                                              SnapsToDevicePixels="{TemplateBinding SnapsToDevicePixels}"/>
                        </Border>
                        <ControlTemplate.Triggers>
                            <Trigger Property="IsMouseOver" Value="True">
                                <Setter Property="Opacity" Value="0.9"/>
                            </Trigger>
                            <Trigger Property="IsPressed" Value="True">
                                <Setter Property="Opacity" Value="0.75"/>
                            </Trigger>
                            <Trigger Property="IsEnabled" Value="False">
                                <Setter TargetName="border" Property="Background" Value="#20202A"/>
                                <Setter TargetName="border" Property="BorderBrush" Value="#2D2D3B"/>
                                <Setter Property="Foreground" Value="#64748B"/>
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>
    </Window.Resources>
    <Grid Margin="16">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/> <!-- 0: Title -->
            <RowDefinition Height="Auto"/> <!-- 1: Files -->
            <RowDefinition Height="Auto"/> <!-- 2: Config -->
            <RowDefinition Height="Auto"/> <!-- 3: Actions -->
            <RowDefinition Height="Auto"/> <!-- 4: Progress -->
            <RowDefinition Height="*"/>    <!-- 5: Log Output -->
        </Grid.RowDefinitions>

        <TextBlock Grid.Row="0" Text="Importance Matrix Calculation (GPU Accelerated)" FontSize="16" FontWeight="Bold" Foreground="#38BDF8" Margin="0,0,0,12"/>

        <Border Grid.Row="1" Background="#20202A" CornerRadius="6" Padding="12" Margin="0,0,0,10" BorderBrush="#2D2D3B" BorderThickness="1">
            <Grid>
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                </Grid.RowDefinitions>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="130"/>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>

                <TextBlock Grid.Row="0" Grid.Column="0" Text="llama-imatrix:" VerticalAlignment="Center" Margin="0,0,0,8" ToolTip="Path to llama-imatrix executable binary"/>
                <TextBox Grid.Row="0" Grid.Column="1" Name="DlgTxtBin" Height="28" Margin="0,0,8,8" ToolTip="Full path to llama-imatrix.exe"/>
                <Button Grid.Row="0" Grid.Column="2" Name="DlgBtnBrowseBin" Content="Browse..." Width="80" Height="28" Margin="0,0,0,8" ToolTip="Browse filesystem for llama-imatrix.exe"/>

                <TextBlock Grid.Row="1" Grid.Column="0" Text="Source Model:" VerticalAlignment="Center" Margin="0,0,0,8" ToolTip="Source GGUF model used to evaluate token activations"/>
                <TextBox Grid.Row="1" Grid.Column="1" Name="DlgTxtSource" Height="28" Margin="0,0,8,8" ToolTip="Path to source model (unquantized F16/BF16 or high-precision quant)"/>
                <Button Grid.Row="1" Grid.Column="2" Name="DlgBtnBrowseSource" Content="Browse..." Width="80" Height="28" Margin="0,0,0,8" ToolTip="Browse filesystem for source GGUF model"/>

                <TextBlock Grid.Row="2" Grid.Column="0" Text="Calibration (.txt):" VerticalAlignment="Center" Margin="0,0,0,8" ToolTip="Plain text training/calibration dataset file"/>
                <TextBox Grid.Row="2" Grid.Column="1" Name="DlgTxtCalibration" Height="28" Margin="0,0,8,8" ToolTip="Calibration dataset (.txt) containing diverse domain or conversational text"/>
                <Button Grid.Row="2" Grid.Column="2" Name="DlgBtnBrowseCalibration" Content="Browse..." Width="80" Height="28" Margin="0,0,0,8" ToolTip="Select calibration text file (.txt)"/>

                <TextBlock Grid.Row="3" Grid.Column="0" Text="Output Matrix:" VerticalAlignment="Center" ToolTip="Destination path for generated importance matrix file"/>
                <TextBox Grid.Row="3" Grid.Column="1" Name="DlgTxtOutput" Height="28" Margin="0,0,8,0" ToolTip="Output file path (.gguf) for importance matrix"/>
                <Button Grid.Row="3" Grid.Column="2" Name="DlgBtnBrowseOutput" Content="Save As..." Width="80" Height="28" ToolTip="Choose destination path for the importance matrix"/>
            </Grid>
        </Border>

        <Border Grid.Row="2" Background="#20202A" CornerRadius="6" Padding="12" Margin="0,0,0,10" BorderBrush="#2D2D3B" BorderThickness="1">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="Auto"/>
                    <ColumnDefinition Width="65"/>
                    <ColumnDefinition Width="Auto"/>
                    <ColumnDefinition Width="65"/>
                    <ColumnDefinition Width="Auto"/>
                    <ColumnDefinition Width="65"/>
                    <ColumnDefinition Width="Auto"/>
                    <ColumnDefinition Width="70"/>
                </Grid.ColumnDefinitions>

                <TextBlock Grid.Column="0" Text="GPU Layers (-ngl):" VerticalAlignment="Center" Margin="0,0,6,0" ToolTip="Number of layers to offload to GPU/VRAM. 99 offloads all layers for maximum GPU acceleration."/>
                <TextBox Grid.Column="1" Name="DlgTxtNgl" Text="99" Height="28" Margin="0,0,12,0" ToolTip="GPU layer offload count (-ngl). Use 99 for full GPU offload."/>

                <TextBlock Grid.Column="2" Text="Context (-c):" VerticalAlignment="Center" Margin="0,0,6,0" ToolTip="Context window length (-c) for processing tokens during calibration"/>
                <TextBox Grid.Column="3" Name="DlgTxtContext" Text="2048" Height="28" Margin="0,0,12,0" ToolTip="Context window length (default: 2048)"/>

                <TextBlock Grid.Column="4" Text="Chunks:" VerticalAlignment="Center" Margin="0,0,6,0" ToolTip="Number of text chunks to process from the dataset"/>
                <TextBox Grid.Column="5" Name="DlgTxtChunks" Text="64" Height="28" Margin="0,0,12,0" ToolTip="Calibration chunk count (default: 64; higher increases accuracy but takes longer)"/>

                <TextBlock Grid.Column="6" Text="Threads:" VerticalAlignment="Center" Margin="0,0,6,0" ToolTip="CPU threads to use during imatrix computation (0 = auto)"/>
                <TextBox Grid.Column="7" Name="DlgTxtThreads" Text="0" Height="28" ToolTip="Thread count (-t); 0 for automatic CPU core detection"/>
            </Grid>
        </Border>

        <Grid Grid.Row="3" Margin="0,0,0,8">
            <Grid.ColumnDefinitions>
                <ColumnDefinition Width="*"/>
                <ColumnDefinition Width="Auto"/>
                <ColumnDefinition Width="Auto"/>
                <ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>
            <TextBlock Grid.Column="0" Name="DlgLblStatus" Text="Ready to compute importance matrix" VerticalAlignment="Center" Foreground="#38BDF8" FontWeight="SemiBold" ToolTip="Status of importance matrix calculation"/>
            <Button Grid.Column="1" Name="DlgBtnCancel" Content="Abort" Width="80" Height="32" Margin="0,0,8,0" Background="#7F1D1D" BorderBrush="#991B1B" IsEnabled="False" ToolTip="Abort running imatrix calculation"/>
            <Button Grid.Column="2" Name="DlgBtnStart" Content="Start Calculation" Width="130" Height="32" Margin="0,0,8,0" Background="#0D9488" BorderBrush="#14B8A6" FontWeight="SemiBold" ToolTip="Start GPU-accelerated importance matrix calculation"/>
            <Button Grid.Column="3" Name="DlgBtnApply" Content="Apply &amp; Close" Width="110" Height="32" Background="#2563EB" BorderBrush="#3B82F6" IsEnabled="False" ToolTip="Set generated matrix as active imatrix and close dialog"/>
        </Grid>

        <!-- 4. Progress Bar -->
        <Grid Grid.Row="4" Margin="0,0,0,8">
            <ProgressBar Name="DlgPrgBar" Height="16" Minimum="0" Maximum="100" Value="0" Background="#1E1E28" Foreground="#0D9488" BorderBrush="#2D2D3B" BorderThickness="1"/>
            <TextBlock Name="DlgLblProgressText" Text="Ready" HorizontalAlignment="Center" VerticalAlignment="Center" FontSize="10" Foreground="#F0F0F5" FontWeight="SemiBold"/>
        </Grid>

        <Border Grid.Row="5" Background="#14141B" CornerRadius="6" BorderBrush="#2D2D3B" BorderThickness="1" Padding="8">
            <TextBox Name="DlgTxtLog" Background="Transparent" Foreground="#E2E8F0" BorderThickness="0"
                     FontFamily="Consolas, Courier New, monospace" FontSize="11"
                     IsReadOnly="True" VerticalScrollBarVisibility="Auto" HorizontalScrollBarVisibility="Auto"
                     AcceptsReturn="True" TextWrapping="NoWrap"/>
        </Border>
    </Grid>
</Window>
"@

    $dialogReader = New-Object System.Xml.XmlNodeReader ([xml]$dialogXaml)
    $dialog = [System.Windows.Markup.XamlReader]::Load($dialogReader)
    $dialog.Owner = $OwnerWindow
    $dialog.Add_SourceInitialized({
        try {
            $helper = New-Object System.Windows.Interop.WindowInteropHelper($dialog)
            $src = [System.Windows.Interop.HwndSource]::FromHwnd($helper.Handle)
            if ($src -and $src.CompositionTarget) {
                $src.CompositionTarget.RenderMode = [System.Windows.Interop.RenderMode]::SoftwareOnly
            }
        } catch {}
    })

    # Dialog Controls
    $dlgTxtBin = $dialog.FindName("DlgTxtBin")
    $dlgBtnBrowseBin = $dialog.FindName("DlgBtnBrowseBin")
    $dlgTxtSource = $dialog.FindName("DlgTxtSource")
    $dlgBtnBrowseSource = $dialog.FindName("DlgBtnBrowseSource")
    $dlgTxtCalibration = $dialog.FindName("DlgTxtCalibration")
    $dlgBtnBrowseCalibration = $dialog.FindName("DlgBtnBrowseCalibration")
    $dlgTxtOutput = $dialog.FindName("DlgTxtOutput")
    $dlgBtnBrowseOutput = $dialog.FindName("DlgBtnBrowseOutput")
    $dlgTxtNgl = $dialog.FindName("DlgTxtNgl")
    $dlgTxtContext = $dialog.FindName("DlgTxtContext")
    $dlgTxtChunks = $dialog.FindName("DlgTxtChunks")
    $dlgTxtThreads = $dialog.FindName("DlgTxtThreads")
    $dlgLblStatus = $dialog.FindName("DlgLblStatus")
    $dlgBtnCancel = $dialog.FindName("DlgBtnCancel")
    $dlgBtnStart = $dialog.FindName("DlgBtnStart")
    $dlgBtnApply = $dialog.FindName("DlgBtnApply")
    $dlgTxtLog = $dialog.FindName("DlgTxtLog")
    $dlgPrgBar = $dialog.FindName("DlgPrgBar")
    $dlgLblProgressText = $dialog.FindName("DlgLblProgressText")

    $dlgTxtBin.Text = $defaultImatrixBin
    $dlgTxtSource.Text = $CurrentSource
    $dlgTxtCalibration.Text = $CurrentCalibration
    $dlgTxtOutput.Text = $defaultImatrixOut

    $script:DlgRunner = $null
    $script:DlgTimer = $null
    $script:DlgTargetOutput = $null
    $script:GeneratedImatrixResult = $null

    # Dialog Browse Events
    $dlgBtnBrowseBin.Add_Click({
        $fbd = New-Object System.Windows.Forms.OpenFileDialog
        $fbd.Title = "Select llama-imatrix Executable"
        $fbd.Filter = "Executable files (*.exe)|*.exe|All files (*.*)|*.*"
        if ($dlgTxtBin.Text -and (Test-Path $dlgTxtBin.Text)) {
            $fbd.InitialDirectory = Split-Path -Parent $dlgTxtBin.Text
        }
        if ($fbd.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            $dlgTxtBin.Text = $fbd.FileName
            Save-GuiConfig -ImatrixExe $fbd.FileName
        }
    })

    $dlgTxtBin.Add_LostFocus({
        $trimmed = $dlgTxtBin.Text.Trim()
        if ($trimmed -and (Test-Path $trimmed)) {
            Save-GuiConfig -ImatrixExe $trimmed
        }
    })

    $dlgBtnBrowseSource.Add_Click({
        $fbd = New-Object System.Windows.Forms.OpenFileDialog
        $fbd.Title = "Select Source GGUF Model"
        $fbd.Filter = "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*"
        if ($dlgTxtSource.Text -and (Test-Path $dlgTxtSource.Text)) {
            $fbd.InitialDirectory = Split-Path -Parent $dlgTxtSource.Text
        }
        if ($fbd.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            $dlgTxtSource.Text = $fbd.FileName
            $dlgTxtOutput.Text = Get-SuggestedImatrixPath -SourcePath $fbd.FileName
            $splitCheck = Test-GgufSplitsComplete -FilePath $fbd.FileName
            if (-not $splitCheck.Complete) {
                $missingNames = $splitCheck.Missing -join "`r`n  - "
                [System.Windows.MessageBox]::Show("Achtung: Dies ist ein mehrteiliges GGUF-Modell, aber folgende Split-Dateien fehlen im Ordner:`r`n  - $missingNames`r`n`r`nBitte laden Sie alle Teile vor Beginn der Berechnung herunter.", "Unvollstaendiges Modell", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning)
            }
        }
    })

    $dlgBtnBrowseCalibration.Add_Click({
        $fbd = New-Object System.Windows.Forms.OpenFileDialog
        $fbd.Title = "Select Calibration Dataset (.txt)"
        $fbd.Filter = "Text files (*.txt;*.raw)|*.txt;*.raw|All files (*.*)|*.*"
        if ($dlgTxtCalibration.Text -and (Test-Path $dlgTxtCalibration.Text)) {
            $fbd.InitialDirectory = Split-Path -Parent $dlgTxtCalibration.Text
        }
        if ($fbd.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            $dlgTxtCalibration.Text = $fbd.FileName
        }
    })

    $dlgBtnBrowseOutput.Add_Click({
        $sfd = New-Object System.Windows.Forms.SaveFileDialog
        $sfd.Title = "Save Importance Matrix As"
        $sfd.Filter = "GGUF files (*.gguf)|*.gguf|Data files (*.dat)|*.dat|All files (*.*)|*.*"
        if ($dlgTxtOutput.Text) {
            $sfd.FileName = Split-Path -Leaf $dlgTxtOutput.Text
            $parent = Split-Path -Parent $dlgTxtOutput.Text
            if ($parent -and (Test-Path $parent)) { $sfd.InitialDirectory = $parent }
        }
        if ($sfd.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            $dlgTxtOutput.Text = $sfd.FileName
        }
    })

    # Dialog Cancel Event
    $dlgBtnCancel.Add_Click({
        if ($script:DlgRunner -and -not $script:DlgRunner.HasExited) {
            $dlgTxtLog.AppendText("`r`n[ABORT] Cancelling llama-imatrix process...`r`n")
            $script:DlgRunner.Kill()
            if ($script:DlgTimer) { $script:DlgTimer.Stop() }
            $dlgLblStatus.Text = "Calculation aborted."
            $dlgLblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
            $dlgPrgBar.Value = 0
            $dlgLblProgressText.Text = "Aborted"
            $dlgBtnStart.IsEnabled = $true
            $dlgBtnCancel.IsEnabled = $false
        }
    })

    # Dialog Apply Event
    $dlgBtnApply.Add_Click({
        $dialog.Close()
    })

    # Dialog Start Calculation Event
    $dlgBtnStart.Add_Click({
        $bin = $dlgTxtBin.Text.Trim()
        $src = $dlgTxtSource.Text.Trim()
        $cal = $dlgTxtCalibration.Text.Trim()
        $out = $dlgTxtOutput.Text.Trim()

        if (-not $bin -or -not (Test-Path $bin)) {
            [System.Windows.MessageBox]::Show("Valid llama-imatrix executable not found.", "Error", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Error)
            return
        }
        Save-GuiConfig -ImatrixExe $bin
        if (-not $src -or -not (Test-Path $src)) {
            [System.Windows.MessageBox]::Show("Source model file not found.", "Error", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Error)
            return
        }
        $splitCheck = Test-GgufSplitsComplete -FilePath $src
        if (-not $splitCheck.Complete) {
            $missingNames = $splitCheck.Missing -join "`r`n  - "
            [System.Windows.MessageBox]::Show("GGUF Split-Modell unvollstaendig!`r`n`r`nFolgende Split-Teile fehlen im Verzeichnis:`r`n  - $missingNames`r`n`r`nllama-imatrix benoetigt alle Teile eines mehrteiligen Modells. Bitte laden Sie die fehlenden Dateien in denselben Ordner herunter.", "Split-Dateien fehlen", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning)
            return
        }
        if (-not $cal -or -not (Test-Path $cal)) {
            [System.Windows.MessageBox]::Show("Calibration dataset file not found: $cal", "Error", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Error)
            return
        }
        if (-not $out) {
            [System.Windows.MessageBox]::Show("Please specify an output matrix path.", "Error", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Error)
            return
        }

        $outDir = Split-Path -Parent $out
        if ($outDir -and -not (Test-Path $outDir)) {
            New-Item -ItemType Directory -Path $outDir -Force | Out-Null
        }

        $ngl = if ($dlgTxtNgl.Text) { $dlgTxtNgl.Text.Trim() } else { "99" }
        $ctx = if ($dlgTxtContext.Text) { $dlgTxtContext.Text.Trim() } else { "2048" }
        $chunks = if ($dlgTxtChunks.Text) { $dlgTxtChunks.Text.Trim() } else { "64" }
        $threads = if ($dlgTxtThreads.Text) { $dlgTxtThreads.Text.Trim() } else { "0" }

        $nglVal = 0
        [int]::TryParse($ngl, [ref]$nglVal) | Out-Null
        $rocmDevDesc = Set-RocmExecutionEnvironment -GpuLayers $nglVal

        $argsList = New-Object System.Collections.Generic.List[string]
        $argsList.Add("-m"); $argsList.Add($src)
        $argsList.Add("-f"); $argsList.Add($cal)
        $argsList.Add("-o"); $argsList.Add($out)
        $argsList.Add("-ngl"); $argsList.Add($ngl)
        $argsList.Add("-c"); $argsList.Add($ctx)
        $argsList.Add("--chunks"); $argsList.Add($chunks)
        if ([int]$threads -gt 0) {
            $argsList.Add("-t"); $argsList.Add($threads)
        }

        $argString = ($argsList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join " "

        $dlgBtnStart.IsEnabled = $false
        $dlgBtnCancel.IsEnabled = $true
        $dlgBtnApply.IsEnabled = $false
        $dlgLblStatus.Text = "Calculating importance matrix..."
        $dlgLblStatus.Foreground = [System.Windows.Media.Brushes]::Yellow
        $dlgPrgBar.Minimum = 0
        $dlgPrgBar.Maximum = 100
        $dlgPrgBar.Value = 0
        $dlgLblProgressText.Text = "Starting computation..."

        $dlgTxtLog.Clear()
        $dlgTxtLog.AppendText("==================================================`r`n")
        $dlgTxtLog.AppendText(" Calculating Importance Matrix (llama-imatrix)`r`n")
        $dlgTxtLog.AppendText("==================================================`r`n")
        $dlgTxtLog.AppendText("Binary:     $bin`r`n")
        $dlgTxtLog.AppendText("Source:     $src`r`n")
        $dlgTxtLog.AppendText("Dataset:    $cal`r`n")
        $dlgTxtLog.AppendText("Output:     $out`r`n")
        $dlgTxtLog.AppendText("GPU Layers: $ngl`r`n")
        $dlgTxtLog.AppendText("ROCm Mode:  $rocmDevDesc`r`n")
        $dlgTxtLog.AppendText("Context:    $ctx`r`n")
        $dlgTxtLog.AppendText("Chunks:     $chunks`r`n")
        $dlgTxtLog.AppendText("Command:    `"$bin`" $argString`r`n")
        $dlgTxtLog.AppendText("==================================================`r`n`r`n")

        $script:DlgTargetOutput = $out
        $script:DlgRunner = New-Object LlamaAsyncProcessRunner

        if ($script:DlgTimer) { $script:DlgTimer.Stop() }
        $script:DlgTimer = New-Object System.Windows.Threading.DispatcherTimer
        $script:DlgTimer.Interval = [TimeSpan]::FromMilliseconds(50)
        $script:DlgTimer.Add_Tick({
            $runner = $script:DlgRunner
            if ($null -eq $runner) { return }

            $line = $null
            $hasNew = $false
            while ($runner.OutputLines.TryDequeue([ref]$line)) {
                $hasNew = $true
                $dlgTxtLog.AppendText($line + "`r`n")

                if ($line -match 'computing over\s+(\d+)\s+chunks') {
                    $totChunks = [int]$Matches[1]
                    $dlgPrgBar.Maximum = $totChunks
                    $dlgPrgBar.Value = 0
                    $dlgLblProgressText.Text = "0 / $totChunks chunks (0%)"
                } elseif ($line -match 'processing chunk\s+(\d+)\s*[/:]\s*(\d+)' -or $line -match '\[\s*(\d+)\s*/\s*(\d+)\s*\]') {
                    $curChunk = [int]$Matches[1]
                    $totChunks = [int]$Matches[2]
                    if ($totChunks -gt 0) {
                        $pct = [math]::Round(($curChunk / $totChunks) * 100)
                        $dlgPrgBar.Maximum = $totChunks
                        $dlgPrgBar.Value = $curChunk
                        $dlgLblProgressText.Text = "$pct% ($curChunk / $totChunks chunks)"
                        $dlgLblStatus.Text = "Computing: $pct% ($curChunk/$totChunks)"
                    }
                }
            }
            if ($hasNew) {
                $dlgTxtLog.ScrollToEnd()
            }

            if ($runner.HasExited -and $runner.OutputLines.IsEmpty) {
                $script:DlgTimer.Stop()
                $exitCode = $runner.ExitCode
                $targetFile = $script:DlgTargetOutput
                $dlgTxtLog.AppendText("`r`n--------------------------------------------------`r`n")
                if ($exitCode -eq 0 -and ($targetFile -and (Test-Path $targetFile))) {
                    $mb = [math]::Round((Get-Item $targetFile).Length / 1MB, 2)
                    $dlgTxtLog.AppendText("[SUCCESS] Importance matrix calculation complete!`r`n")
                    $dlgTxtLog.AppendText("Created: $targetFile ($mb MB)`r`n")
                    $dlgLblStatus.Text = "Finished: $mb MB generated"
                    $dlgLblStatus.Foreground = [System.Windows.Media.Brushes]::LightGreen
                    $dlgPrgBar.Value = $dlgPrgBar.Maximum
                    $dlgLblProgressText.Text = "100% Complete"
                    $script:GeneratedImatrixResult = $targetFile
                    $dlgBtnApply.IsEnabled = $true
                } else {
                    $dlgTxtLog.AppendText("[ERROR] Calculation failed with exit code $exitCode.`r`n")
                    $dlgLblStatus.Text = "Failed with exit code $exitCode"
                    $dlgLblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
                    $dlgLblProgressText.Text = "Failed"
                }
                $dlgBtnStart.IsEnabled = $true
                $dlgBtnCancel.IsEnabled = $false
            }
        })

        try {
            $binDir = Split-Path -Parent $bin
            $started = $script:DlgRunner.Start($bin, $argString, $binDir)
            if ($started) {
                $script:DlgTimer.Start()
            } else {
                $dlgTxtLog.AppendText("[ERROR] Failed to start llama-imatrix process.`r`n")
                $dlgLblStatus.Text = "Execution failed"
                $dlgLblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
                $dlgLblProgressText.Text = "Failed"
                $dlgBtnStart.IsEnabled = $true
                $dlgBtnCancel.IsEnabled = $false
            }
        } catch {
            $dlgTxtLog.AppendText("[ERROR] Failed to start process: $($_.Exception.Message)`r`n")
            $dlgLblStatus.Text = "Execution failed"
            $dlgLblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
            $dlgLblProgressText.Text = "Failed"
            $dlgBtnStart.IsEnabled = $true
            $dlgBtnCancel.IsEnabled = $false
        }
    })

    # Show modal dialog
    $dialog.ShowDialog() | Out-Null

    # Clean up process if still somehow running when dialog closed
    if ($script:DlgTimer) { $script:DlgTimer.Stop() }
    if ($script:DlgRunner -and -not $script:DlgRunner.HasExited) {
        $script:DlgRunner.Kill()
    }

    return $script:GeneratedImatrixResult
}

# Event: Generate Imatrix (launch dialog)
$btnCreateImatrix.Add_Click({
    $calInitial = if ($InitialCalibration) { $InitialCalibration } else { "" }
    $genImatrix = Show-ImatrixDialog -OwnerWindow $window -CurrentSource $txtSourceModel.Text -CurrentBin $txtExePath.Text -CurrentCalibration $calInitial
    if ($genImatrix) {
        $txtImatrix.Text = $genImatrix
        $txtLog.AppendText("`r`n[IMATRIX] Configured importance matrix: $genImatrix`r`n")
    }
})

# Event: Cancel Process
$btnCancel.Add_Click({
    if ($script:RunningRunner -and -not $script:RunningRunner.HasExited) {
        $txtLog.AppendText("`r`n[ABORT] Cancelling quantization process...`r`n")
        $script:RunningRunner.Kill()
        if ($script:QuantTimer) { $script:QuantTimer.Stop() }
        $lblStatus.Text = "Cancelled"
        $lblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
        $prgBar.Value = 0
        $lblProgressText.Text = "Cancelled"
        $btnStart.IsEnabled = $true
        $btnCancel.IsEnabled = $false
    }
})

# Event: Start Quantization
$btnStart.Add_Click({
    # Validation
    $exe = $txtExePath.Text.Trim()
    if (-not (Test-Path $exe)) {
        [System.Windows.MessageBox]::Show("Quantization executable not found!`nPlease select a valid llama-quantize.exe.", "Executable Missing", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning)
        return
    }
    Save-GuiConfig -QuantizeExe $exe

    $source = $txtSourceModel.Text.Trim()
    if (-not (Test-Path $source)) {
        [System.Windows.MessageBox]::Show("Source model file not found!`nPlease select an existing GGUF model.", "Source Missing", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning)
        return
    }
    $splitCheck = Test-GgufSplitsComplete -FilePath $source
    if (-not $splitCheck.Complete) {
        $missingNames = $splitCheck.Missing -join "`r`n  - "
        [System.Windows.MessageBox]::Show("GGUF Split-Modell unvollstaendig!`r`n`r`nFolgende Split-Teile fehlen im Verzeichnis:`r`n  - $missingNames`r`n`r`nllama-quantize benoetigt alle Teile eines mehrteiligen Modells. Bitte laden Sie die fehlenden Dateien in denselben Ordner herunter.", "Split-Dateien fehlen", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning)
        return
    }

    $output = $txtOutputModel.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($output)) {
        [System.Windows.MessageBox]::Show("Output model path cannot be empty!", "Output Missing", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning)
        return
    }

    $preset = Get-SelectedPresetTag
    $imatrix = $txtImatrix.Text.Trim()
    if (-not [string]::IsNullOrWhiteSpace($imatrix) -and -not (Test-Path $imatrix)) {
        [System.Windows.MessageBox]::Show("Specified Imatrix file does not exist: $imatrix", "Imatrix Not Found", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning)
        return
    }

    $allowRequant = $chkAllowRequantize.IsChecked -eq $true
    $threads = Get-SelectedThreads

    # Create destination directory if needed
    $outDir = Split-Path -Parent $output
    if ($outDir -and -not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    }

    # Build arguments
    $argsArray = Build-QuantizeArguments -Source $source -Output $output -Preset $preset -Imatrix $imatrix -AllowRequantize $allowRequant -Threads $threads
    $argString = ($argsArray | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join " "

    # UI updates for running state
    $btnStart.IsEnabled = $false
    $btnCancel.IsEnabled = $true
    $lblStatus.Text = "Quantizing to $preset..."
    $lblStatus.Foreground = [System.Windows.Media.Brushes]::Yellow
    $prgBar.Minimum = 0
    $prgBar.Maximum = 100
    $prgBar.Value = 0
    $lblProgressText.Text = "Starting..."
    $txtLog.Clear()
    $txtLog.AppendText("==================================================`r`n")
    $txtLog.AppendText(" ROCmFPX & TurboQuant Model Quantizer`r`n")
    $txtLog.AppendText("==================================================`r`n")
    $txtLog.AppendText("Binary:    $exe`r`n")
    $txtLog.AppendText("Source:    $source`r`n")
    $txtLog.AppendText("Output:    $output`r`n")
    $txtLog.AppendText("Preset:    $preset`r`n")
    if ($imatrix) { $txtLog.AppendText("Imatrix:   $imatrix`r`n") }
    if ($allowRequant) { $txtLog.AppendText("Requant:   Allowed`r`n") }
    $txtLog.AppendText("Command:   `"$exe`" $argString`r`n")
    $txtLog.AppendText("==================================================`r`n`r`n")

    $rocmDevDesc = Set-RocmExecutionEnvironment -GpuLayers 0

    $script:QuantOutputFile = $output
    $script:RunningRunner = New-Object LlamaAsyncProcessRunner

    if ($script:QuantTimer) { $script:QuantTimer.Stop() }
    $script:QuantTimer = New-Object System.Windows.Threading.DispatcherTimer
    $script:QuantTimer.Interval = [TimeSpan]::FromMilliseconds(50)
    $script:QuantTimer.Add_Tick({
        $runner = $script:RunningRunner
        if ($null -eq $runner) { return }

        $line = $null
        $hasNew = $false
        while ($runner.OutputLines.TryDequeue([ref]$line)) {
            $hasNew = $true
            $txtLog.AppendText($line + "`r`n")

            # Parse [   1/ 291] tensor progress
            if ($line -match '\[\s*(\d+)\s*/\s*(\d+)\s*\]') {
                $curTensor = [int]$Matches[1]
                $totTensors = [int]$Matches[2]
                if ($totTensors -gt 0) {
                    $pct = [math]::Round(($curTensor / $totTensors) * 100)
                    $prgBar.Maximum = $totTensors
                    $prgBar.Value = $curTensor
                    $lblProgressText.Text = "$pct% ($curTensor / $totTensors tensors)"
                    $lblStatus.Text = "Quantizing: $pct% ($curTensor/$totTensors)"
                }
            }
        }
        if ($hasNew) {
            $txtLog.ScrollToEnd()
        }

        if ($runner.HasExited -and $runner.OutputLines.IsEmpty) {
            $script:QuantTimer.Stop()
            $exitCode = $runner.ExitCode
            $targetFile = $script:QuantOutputFile
            $txtLog.AppendText("`r`n--------------------------------------------------`r`n")
            if ($exitCode -eq 0 -and ($targetFile -and (Test-Path $targetFile))) {
                $sizeBytes = (Get-Item $targetFile).Length
                $sizeMB = [math]::Round($sizeBytes / 1MB, 2)
                $txtLog.AppendText("[SUCCESS] Quantization complete!`r`n")
                $txtLog.AppendText("File: $targetFile ($sizeMB MB)`r`n")
                $lblStatus.Text = "Finished: $sizeMB MB created"
                $lblStatus.Foreground = [System.Windows.Media.Brushes]::LightGreen
                $prgBar.Value = $prgBar.Maximum
                $lblProgressText.Text = "100% Complete"
            } else {
                $txtLog.AppendText("[ERROR] Process exited with code: $exitCode`r`n")
                $lblStatus.Text = "Failed (Exit code: $exitCode)"
                $lblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
                $lblProgressText.Text = "Failed"
            }
            $btnStart.IsEnabled = $true
            $btnCancel.IsEnabled = $false
        }
    })

    try {
        $binDir = Split-Path -Parent $exe
        $started = $script:RunningRunner.Start($exe, $argString, $binDir)
        if ($started) {
            $script:QuantTimer.Start()
        } else {
            $txtLog.AppendText("[ERROR] Failed to start process.`r`n")
            $btnStart.IsEnabled = $true
            $btnCancel.IsEnabled = $false
            $lblStatus.Text = "Start Failed"
            $lblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
            $lblProgressText.Text = "Failed"
        }
    } catch {
        $txtLog.AppendText("[EXCEPTION] $($_.Exception.Message)`r`n")
        $btnStart.IsEnabled = $true
        $btnCancel.IsEnabled = $false
        $lblStatus.Text = "Exception"
        $lblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
        $lblProgressText.Text = "Exception"
    }
})

# Window Closing Event: clean up background processes
$window.Add_Closing({
    if ($script:QuantTimer) { $script:QuantTimer.Stop() }
    if ($script:RunningRunner -and -not $script:RunningRunner.HasExited) {
        $script:RunningRunner.Kill()
    }
})

# Show the WPF Window
$window.ShowDialog() | Out-Null
