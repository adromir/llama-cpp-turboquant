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
    [string]$InitialPreset = "Q4_0_ROCMFP4_FAST"
)

Set-StrictMode -Off

# Ensure WPF assemblies are loaded
Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase, System.Windows.Forms | Out-Null

# ==============================================================================
# Business Logic
# ==============================================================================

function Find-QuantizeBinary {
    param([string]$ScriptDir)

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

    # Strip existing quant tags if present (e.g., -F16, -BF16, -Q8_0, -Q4_K_M)
    $cleanBase = $baseName -replace "-(BF16|F16|Q8_0|Q4_K_M|Q4_0|Q6_K|Q5_K_M|f16|bf16)$", ""
    $cleanBase = $cleanBase -replace "-(Q[0-9]_[0-9A-Z_]+|tq[0-9]_[0-9a-z]+)$", ""

    $newName = "$cleanBase-$Preset.gguf"
    if ($dir) {
        return Join-Path $dir $newName
    }
    return $newName
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
        </Style>
        <Style TargetType="Button">
            <Setter Property="Background" Value="#2E2E3D"/>
            <Setter Property="Foreground" Value="#FFFFFF"/>
            <Setter Property="BorderBrush" Value="#45455A"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="Padding" Value="12,5"/>
            <Setter Property="Cursor" Value="Hand"/>
        </Style>
        <Style TargetType="ComboBox">
            <Setter Property="Background" Value="#22222C"/>
            <Setter Property="Foreground" Value="#FFFFFF"/>
            <Setter Property="BorderBrush" Value="#383848"/>
            <Setter Property="Padding" Value="6,4"/>
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

                <TextBlock Grid.Row="0" Grid.ColumnSpan="2" Text="Quantization Executable (llama-quantize.exe)" FontWeight="SemiBold" Margin="0,0,0,6"/>
                
                <TextBox Grid.Row="1" Grid.Column="0" Name="TxtExePath" Height="30" Margin="0,0,8,0"/>
                <Button Grid.Row="1" Grid.Column="1" Name="BtnBrowseExe" Content="Browse..." Width="90" Height="30"/>

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
                <TextBlock Grid.Row="0" Grid.Column="0" Text="Source Model:" VerticalAlignment="Center" Margin="0,0,0,10"/>
                <TextBox Grid.Row="0" Grid.Column="1" Name="TxtSourceModel" Height="30" Margin="0,0,8,10"/>
                <Button Grid.Row="0" Grid.Column="2" Name="BtnBrowseSource" Content="Browse..." Width="90" Height="30" Margin="0,0,0,10"/>

                <!-- Preset Selection -->
                <TextBlock Grid.Row="1" Grid.Column="0" Text="Quant Preset:" VerticalAlignment="Center" Margin="0,0,0,10"/>
                <ComboBox Grid.Row="1" Grid.Column="1" Name="CmbPreset" Height="30" Margin="0,0,8,10">
                    <!-- ROCmFP4 -->
                    <ComboBoxItem Content="Q4_0_ROCMFP4_FAST - Fastest 4-Bit FP4 (Recommended)" Tag="Q4_0_ROCMFP4_FAST" IsSelected="True"/>
                    <ComboBoxItem Content="Q4_0_ROCMFP4 - Standard 4-Bit FP4" Tag="Q4_0_ROCMFP4"/>
                    <ComboBoxItem Content="Q4_0_ROCMFP4_COHERENT - Coherent 4-Bit FP4 (Higher Quality)" Tag="Q4_0_ROCMFP4_COHERENT"/>
                    <ComboBoxItem Content="Q4_0_ROCMFP4_STRIX - Tuned for Strix Point iGPU" Tag="Q4_0_ROCMFP4_STRIX"/>
                    <ComboBoxItem Content="Q4_0_ROCMFP4_STRIX_LEAN - Lean Strix Point Format" Tag="Q4_0_ROCMFP4_STRIX_LEAN"/>
                    <!-- ROCmFP3 -->
                    <ComboBoxItem Content="Q3_0_ROCMFPX - 3-Bit FP3 (Fastest 3-Bit)" Tag="Q3_0_ROCMFPX"/>
                    <ComboBoxItem Content="Q3_0_ROCMFPX_AGENT - 3-Bit FP3 Agent (Best with Imatrix)" Tag="Q3_0_ROCMFPX_AGENT"/>
                    <!-- ROCmFP6 -->
                    <ComboBoxItem Content="Q6_0_ROCMFPX - 6-Bit FP6 (High Precision)" Tag="Q6_0_ROCMFPX"/>
                    <ComboBoxItem Content="Q6_0_ROCMFPX_AGENT - 6-Bit FP6 Agent (Near Lossless)" Tag="Q6_0_ROCMFPX_AGENT"/>
                    <ComboBoxItem Content="Q6_0_ROCMFPX_LEAN - 6-Bit FP6 Lean" Tag="Q6_0_ROCMFPX_LEAN"/>
                    <!-- ROCmFP8 -->
                    <ComboBoxItem Content="Q8_0_ROCMFPX - 8-Bit FP8" Tag="Q8_0_ROCMFPX"/>
                    <ComboBoxItem Content="Q8_0_ROCMFPX_AGENT - 8-Bit FP8 Agent" Tag="Q8_0_ROCMFPX_AGENT"/>
                    <!-- ROCm Integer -->
                    <ComboBoxItem Content="Q4_0_ROCMI4 - 4-Bit Integer ROCm" Tag="Q4_0_ROCMI4"/>
                    <!-- TurboQuant Weights -->
                    <ComboBoxItem Content="tq3_1s - TurboQuant 3-Bit (WHT-rotated Lloyd-Max)" Tag="tq3_1s"/>
                    <ComboBoxItem Content="tq4_1s - TurboQuant 4-Bit (WHT-rotated Lloyd-Max)" Tag="tq4_1s"/>
                    <!-- Upstream Standard Formats -->
                    <ComboBoxItem Content="Q4_K_M - Upstream 4-Bit K-Quant" Tag="Q4_K_M"/>
                    <ComboBoxItem Content="Q5_K_M - Upstream 5-Bit K-Quant" Tag="Q5_K_M"/>
                    <ComboBoxItem Content="Q6_K - Upstream 6-Bit K-Quant" Tag="Q6_K"/>
                    <ComboBoxItem Content="Q8_0 - Upstream 8-Bit Standard" Tag="Q8_0"/>
                    <ComboBoxItem Content="IQ3_S - Upstream 3-Bit I-Quant (Needs Imatrix)" Tag="IQ3_S"/>
                </ComboBox>
                <Button Grid.Row="1" Grid.Column="2" Name="BtnSuggestOutput" Content="Auto-Name" Width="90" Height="30" Margin="0,0,0,10" ToolTip="Regenerate output path based on source and preset"/>

                <!-- Output Model -->
                <TextBlock Grid.Row="2" Grid.Column="0" Text="Output Model:" VerticalAlignment="Center" Margin="0,0,0,10"/>
                <TextBox Grid.Row="2" Grid.Column="1" Name="TxtOutputModel" Height="30" Margin="0,0,8,10"/>
                <Button Grid.Row="2" Grid.Column="2" Name="BtnBrowseOutput" Content="Save As..." Width="90" Height="30" Margin="0,0,0,10"/>

                <!-- Importance Matrix -->
                <TextBlock Grid.Row="3" Grid.Column="0" Text="Imatrix (Optional):" VerticalAlignment="Center"/>
                <TextBox Grid.Row="3" Grid.Column="1" Name="TxtImatrix" Height="30" Margin="0,0,8,0"/>
                <Button Grid.Row="3" Grid.Column="2" Name="BtnBrowseImatrix" Content="Browse..." Width="90" Height="30"/>
            </Grid>
        </Border>

        <!-- 4. Options Section -->
        <Border Grid.Row="3" Background="#20202A" CornerRadius="6" Padding="12" Margin="0,0,0,12" BorderBrush="#2D2D3B" BorderThickness="1">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                    <ColumnDefinition Width="80"/>
                </Grid.ColumnDefinitions>

                <CheckBox Grid.Column="0" Name="ChkAllowRequantize" Content="Allow Requantize (--allow-requantize)" ToolTip="Check if source is already quantized (e.g. Q8_0 or Q4_K_M) rather than F16/BF16"/>

                <TextBlock Grid.Column="1" Text="CPU Threads:" VerticalAlignment="Center" Margin="0,0,8,0"/>
                <ComboBox Grid.Column="2" Name="CmbThreads" Height="28">
                    <ComboBoxItem Content="0 (Auto)" Tag="0" IsSelected="True"/>
                    <ComboBoxItem Content="4" Tag="4"/>
                    <ComboBoxItem Content="8" Tag="8"/>
                    <ComboBoxItem Content="12" Tag="12"/>
                    <ComboBoxItem Content="16" Tag="16"/>
                    <ComboBoxItem Content="24" Tag="24"/>
                    <ComboBoxItem Content="32" Tag="32"/>
                </ComboBox>
            </Grid>
        </Border>

        <!-- 5. Actions & Status -->
        <Grid Grid.Row="4" Margin="0,0,0,10">
            <Grid.ColumnDefinitions>
                <ColumnDefinition Width="*"/>
                <ColumnDefinition Width="Auto"/>
                <ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>

            <TextBlock Grid.Column="0" Name="LblStatus" Text="Ready" VerticalAlignment="Center" FontWeight="SemiBold" Foreground="#38BDF8"/>
            <Button Grid.Column="1" Name="BtnCancel" Content="Cancel" Width="100" Height="34" Margin="0,0,10,0" IsEnabled="False" Background="#7F1D1D" BorderBrush="#991B1B"/>
            <Button Grid.Column="2" Name="BtnStart" Content="Start Quantization" Width="160" Height="34" FontWeight="SemiBold" Background="#2563EB" BorderBrush="#3B82F6"/>
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
$chkAllowRequantize   = $window.FindName("ChkAllowRequantize")
$cmbThreads           = $window.FindName("CmbThreads")
$lblStatus            = $window.FindName("LblStatus")
$btnStart             = $window.FindName("BtnStart")
$btnCancel            = $window.FindName("BtnCancel")
$txtLog               = $window.FindName("TxtLog")

# Store running process reference
$script:RunningProcess = $null

# ==============================================================================
# Controller & Event Wiring
# ==============================================================================

# 1. Initialize Executable Detection
$foundExe = Find-QuantizeBinary -ScriptDir $PSScriptRoot
if ($foundExe) {
    $txtExePath.Text = $foundExe.Path
    $lblExeStatus.Text = "Auto-detected ($($foundExe.Origin)): $($foundExe.Path)"
    $lblExeStatus.Foreground = [System.Windows.Media.Brushes]::LightGreen
} else {
    $lblExeStatus.Text = "llama-quantize.exe not found automatically. Please browse and select it."
    $lblExeStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
}

# Apply initial source if provided
if ($InitialSource) {
    $txtSourceModel.Text = $InitialSource
    $selectedPresetTag = $cmbPreset.SelectedItem.Tag
    $txtOutputModel.Text = Get-SuggestedOutputPath -SourcePath $InitialSource -Preset $selectedPresetTag
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
        $lblExeStatus.Text = "Custom executable: $($dlg.FileName)"
        $lblExeStatus.Foreground = [System.Windows.Media.Brushes]::LightSkyBlue
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

# Event: Cancel Process
$btnCancel.Add_Click({
    if ($script:RunningProcess -and -not $script:RunningProcess.HasExited) {
        $txtLog.AppendText("`r`n[ABORT] Cancelling quantization process...`r`n")
        try {
            $script:RunningProcess.Kill()
        } catch {
            $txtLog.AppendText("[WARN] Could not kill process: $($_.Exception.Message)`r`n")
        }
        $lblStatus.Text = "Cancelled"
        $lblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
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

    $source = $txtSourceModel.Text.Trim()
    if (-not (Test-Path $source)) {
        [System.Windows.MessageBox]::Show("Source model file not found!`nPlease select an existing GGUF model.", "Source Missing", [System.Windows.MessageBoxButton]::OK, [System.Windows.MessageBoxImage]::Warning)
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

    # Start process asynchronously with redirected streams
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo.FileName = $exe
    $proc.StartInfo.Arguments = $argString
    $proc.StartInfo.UseShellExecute = $false
    $proc.StartInfo.RedirectStandardOutput = $true
    $proc.StartInfo.RedirectStandardError = $true
    $proc.StartInfo.CreateNoWindow = $true
    $proc.EnableRaisingEvents = $true

    $script:RunningProcess = $proc

    # Live output handler
    $outputHandler = {
        param($sender, $e)
        if ($e.Data) {
            $txtLog.Dispatcher.Invoke([Action]{
                $txtLog.AppendText($e.Data + "`r`n")
                $txtLog.ScrollToEnd()
            })
        }
    }

    $proc.add_OutputDataReceived($outputHandler)
    $proc.add_ErrorDataReceived($outputHandler)

    # Process exit handler
    $proc.add_Exited({
        $exitCode = $script:RunningProcess.ExitCode
        $txtLog.Dispatcher.Invoke([Action]{
            $txtLog.AppendText("`r`n--------------------------------------------------`r`n")
            if ($exitCode -eq 0 -and (Test-Path $output)) {
                $sizeBytes = (Get-Item $output).Length
                $sizeMB = [math]::Round($sizeBytes / 1MB, 2)
                $txtLog.AppendText("[SUCCESS] Quantization complete!`r`n")
                $txtLog.AppendText("File: $output ($sizeMB MB)`r`n")
                $lblStatus.Text = "Finished: $sizeMB MB created"
                $lblStatus.Foreground = [System.Windows.Media.Brushes]::LightGreen
            } else {
                $txtLog.AppendText("[ERROR] Process exited with code: $exitCode`r`n")
                $lblStatus.Text = "Failed (Exit code: $exitCode)"
                $lblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
            }
            $btnStart.IsEnabled = $true
            $btnCancel.IsEnabled = $false
        })
    })

    try {
        $started = $proc.Start()
        if ($started) {
            $proc.BeginOutputReadLine()
            $proc.BeginErrorReadLine()
        } else {
            $txtLog.AppendText("[ERROR] Failed to start process.`r`n")
            $btnStart.IsEnabled = $true
            $btnCancel.IsEnabled = $false
            $lblStatus.Text = "Start Failed"
            $lblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
        }
    } catch {
        $txtLog.AppendText("[EXCEPTION] $($_.Exception.Message)`r`n")
        $btnStart.IsEnabled = $true
        $btnCancel.IsEnabled = $false
        $lblStatus.Text = "Exception"
        $lblStatus.Foreground = [System.Windows.Media.Brushes]::Salmon
    }
})

# Show the WPF Window
$window.ShowDialog() | Out-Null
