# build_to_flutter.ps1
# Build the Vite viewer and sync the output to Flutter assets.
# Usage: run .\build_to_flutter.ps1 in my-3dgs-viewer/

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Building Vite project..." -ForegroundColor Cyan
Write-Host "====================================="

npm run build-only
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  Syncing to Flutter assets (webgl)..." -ForegroundColor Cyan
Write-Host "====================================="

$scriptDir = $PSScriptRoot
$target = Join-Path $scriptDir "..\..\app\assets\webgl"

Remove-Item -Recurse -Force "$target\assets" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$target\models" -ErrorAction SilentlyContinue
Remove-Item -Force "$target\index.html" -ErrorAction SilentlyContinue
Remove-Item -Force "$target\favicon.ico" -ErrorAction SilentlyContinue

Copy-Item -Recurse "$scriptDir\dist\assets" "$target\assets"
Copy-Item "$scriptDir\dist\index.html" "$target\index.html"
if (Test-Path "$scriptDir\dist\models") {
    Copy-Item -Recurse "$scriptDir\dist\models" "$target\models"
}
if (Test-Path "$scriptDir\dist\favicon.ico") {
    Copy-Item "$scriptDir\dist\favicon.ico" "$target\favicon.ico"
}

if (-not (Test-Path "$target\models")) {
    New-Item -ItemType Directory -Path "$target\models" | Out-Null
}
New-Item -ItemType File -Path "$target\models\.gitkeep" -Force | Out-Null

Write-Host "   Synced: $target" -ForegroundColor Gray
Write-Host ""
Write-Host "Done! Flutter app will use the new production build." -ForegroundColor Green
Write-Host "Run 'flutter run' in the app/ directory." -ForegroundColor Green
