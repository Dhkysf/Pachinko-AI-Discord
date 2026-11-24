param(
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
$PythonwExe = Join-Path $Root ".venv\Scripts\pythonw.exe"
$EntryScript = Join-Path $Root "Pachinko.pyw"
$IconPath = Join-Path $Root "app\pachinko.ico"

if (-not (Test-Path $PythonExe)) {
    throw ".venv\Scripts\python.exe not found. Run setup.bat first."
}

& $PythonExe -m app.tray --write-ico $IconPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to generate app\pachinko.ico"
}

$Targets = @()
if ($Destination -ne "") {
    if (-not (Test-Path $Destination)) {
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    }
    $Targets += (Join-Path $Destination "Pachinko.lnk")
} else {
    $Targets += (Join-Path $Root "Pachinko.lnk")
    $Desktop = [Environment]::GetFolderPath("Desktop")
    if ($Desktop) {
        $Targets += (Join-Path $Desktop "Pachinko.lnk")
    }
}

$WshShell = New-Object -ComObject WScript.Shell
foreach ($LinkPath in $Targets) {
    $Shortcut = $WshShell.CreateShortcut($LinkPath)
    $Shortcut.TargetPath = $PythonwExe
    $Shortcut.Arguments = "`"$EntryScript`""
    $Shortcut.WorkingDirectory = $Root
    $Shortcut.IconLocation = $IconPath
    $Shortcut.Description = "Pachinko Control"
    $Shortcut.Save()
    Write-Output "Created shortcut: $LinkPath"
}
