# build_to_flutter.ps1
# 将 Vite 构建产物同步到 Flutter assets
# 用法：在 my-3dgs-viewer/ 目录下执行 .\build_to_flutter.ps1

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
Write-Host "  Syncing to Flutter assets..." -ForegroundColor Cyan
Write-Host "====================================="

# 使用脚本自身所在目录计算绝对路径，避免工作目录不同导致的路径错误
$scriptDir = $PSScriptRoot
$target = Join-Path $scriptDir "..\..\app\assets\webgl"

# 清空目标目录中的 assets/ index.html favicon.ico（保留 models/ 本地 demo 数据）
Remove-Item -Recurse -Force "$target\assets" -ErrorAction SilentlyContinue
Remove-Item -Force "$target\index.html" -ErrorAction SilentlyContinue
Remove-Item -Force "$target\favicon.ico" -ErrorAction SilentlyContinue

# 复制新的构建产物
Copy-Item -Recurse "$scriptDir\dist\assets" "$target\assets"
Copy-Item "$scriptDir\dist\index.html" "$target\index.html"
Copy-Item "$scriptDir\dist\favicon.ico" "$target\favicon.ico"

Write-Host ""
Write-Host "✅ Done! Flutter app will use the new production build." -ForegroundColor Green
Write-Host "   Target: $target" -ForegroundColor Gray
Write-Host "   Run 'flutter run' in the app/ directory." -ForegroundColor Green
