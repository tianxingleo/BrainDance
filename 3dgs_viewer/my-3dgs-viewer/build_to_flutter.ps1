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

$target = "../../../../app/assets/webgl"

# 清空目标目录（保留目录本身）
Get-ChildItem -Path $target -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

# 复制新的构建产物
Copy-Item -Recurse dist/* $target/

Write-Host ""
Write-Host "✅ Done! Flutter app will use the new production build." -ForegroundColor Green
Write-Host "   Run 'flutter run' in the app/ directory." -ForegroundColor Green
