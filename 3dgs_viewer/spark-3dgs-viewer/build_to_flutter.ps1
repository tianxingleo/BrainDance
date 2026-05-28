# build_to_flutter.ps1
# Build the Spark Vite viewer and sync the output to Flutter assets (webgl_spark only).
# Usage: run .\build_to_flutter.ps1 in spark-3dgs-viewer/

$scriptDir = $PSScriptRoot
Push-Location $scriptDir

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Building Spark Vite project..." -ForegroundColor Cyan
Write-Host "====================================="

npm run build-only
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Syncing to Flutter assets (webgl_spark)..." -ForegroundColor Cyan
Write-Host "====================================="

$target = Join-Path $scriptDir "..\..\app\assets\webgl_spark"

Remove-Item -Recurse -Force "$target\assets" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$target\models" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$target\targets" -ErrorAction SilentlyContinue
Remove-Item -Force "$target\index.html" -ErrorAction SilentlyContinue
Remove-Item -Force "$target\favicon.ico" -ErrorAction SilentlyContinue

Copy-Item -Recurse "$scriptDir\dist\assets" "$target\assets"
Copy-Item "$scriptDir\dist\index.html" "$target\index.html"
if (Test-Path "$scriptDir\dist\models") {
    Copy-Item -Recurse "$scriptDir\dist\models" "$target\models"
}
if (Test-Path "$scriptDir\dist\targets") {
    Copy-Item -Recurse "$scriptDir\dist\targets" "$target\targets"
}
if (Test-Path "$scriptDir\dist\favicon.ico") {
    Copy-Item "$scriptDir\dist\favicon.ico" "$target\favicon.ico"
}

if (-not (Test-Path "$target\models")) {
    New-Item -ItemType Directory -Path "$target\models" | Out-Null
}
if (-not (Test-Path "$target\targets")) {
    New-Item -ItemType Directory -Path "$target\targets" | Out-Null
}
New-Item -ItemType File -Path "$target\models\.gitkeep" -Force | Out-Null
New-Item -ItemType File -Path "$target\targets\.gitkeep" -Force | Out-Null

Write-Host "   Synced: $target" -ForegroundColor Gray
Write-Host ""
Write-Host "Done! Flutter app will use the new Spark production build." -ForegroundColor Green
Write-Host "Run 'flutter run' in the app/ directory." -ForegroundColor Green

Pop-Location
