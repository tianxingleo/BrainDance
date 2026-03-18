import urllib.request
import json
import urllib.error

url = 'https://supabase.tianxingleo.top/storage/v1/object/public/braindance-assets/test1/scene_party_001/output/webgl_poses.json'

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    with urllib.request.urlopen(req) as response:
        print("OK", response.status)
        data = json.loads(response.read().decode('utf-8'))
        print("Total frames:", len(data.get('frames', [])))
        if len(data.get('frames', [])) > 0:
            print("First image_url:", data['frames'][0]['image_url'])
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code, e.reason)
    print(e.read().decode())
