$ErrorActionPreference = "Stop"

$url = "https://supabase.tianxingleo.top"
$key = $env:SUPABASE_ANON_KEY
if (-not $key) { $key = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH" }

$bucket = "braindance-assets"
$headers = @{
    "apikey" = $key
    "Authorization" = "Bearer $key"
}

# Download json
$jsonUrl = "$url/storage/v1/object/public/$bucket/test1/scene_party_001/output/webgl_poses.json"
Write-Host "Downloading $jsonUrl"
$response = Invoke-RestMethod -Uri $jsonUrl -Headers @{"User-Agent"="Mozilla/5.0"}
$data = $response

# Create tiny dummy jpeg (base64)
$dummyBase64 = "/9j/4AAQSkZJRgABAQEASABIAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAEAAAAAAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
$dummyBytes = [System.Convert]::FromBase64String($dummyBase64)
[IO.File]::WriteAllBytes("dummy.jpg", $dummyBytes)

$tags = @("汽车正面特写", "侧面全景", "俯视特写", "房间全景", "细节近景", "汽车侧面")

for ($i = 0; $i -lt 6; $i++) {
    $frame = $data.frames[$i]
    if ($frame.PSObject.Properties.Match("tag").Count -eq 0) {
        $frame | Add-Member -MemberType NoteProperty -Name "tag" -Value $tags[$i % $tags.Count]
    } else {
        $frame.tag = $tags[$i % $tags.Count]
    }
    $remoteImgPath = "test1/scene_party_001/output/images/$($frame.id)"
    Write-Host "Uploading $remoteImgPath"
    
    $uploadUrl = "$url/storage/v1/object/$bucket/$remoteImgPath"
    $uploadHeaders = @{
        "apikey" = $key
        "Authorization" = "Bearer $key"
        "x-upsert" = "true"
    }
    
    Invoke-WebRequest -Uri $uploadUrl -Method Post -Headers $uploadHeaders -ContentType "image/jpeg" -InFile "dummy.jpg" | Out-Null
}

for ($i = 6; $i -lt $data.frames.Count; $i++) {
    if ($data.frames[$i].PSObject.Properties.Match("tag").Count -gt 0) {
        $data.frames[$i].PSObject.Properties.Remove("tag")
    }
}

$jsonOutput = $data | ConvertTo-Json -Depth 10 -Compress
[IO.File]::WriteAllText("webgl_poses_fixed.json", $jsonOutput)

Write-Host "Uploading webgl_poses.json"
$uploadUrl = "$url/storage/v1/object/$bucket/test1/scene_party_001/output/webgl_poses.json"
Invoke-WebRequest -Uri $uploadUrl -Method Post -Headers $uploadHeaders -ContentType "application/json" -InFile "webgl_poses_fixed.json" | Out-Null

Write-Host "Done!"
