import urllib.request
import json
import urllib.error
key = 'sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH'
req = urllib.request.Request(
    'https://supabase.tianxingleo.top/rest/v1/memory_poses?select=id,model_id,image_name,tag',
    headers={'apikey': key, 'Authorization': f'Bearer {key}', 'User-Agent': 'Mozilla/5.0'}
)
try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        print(f"Found {len(res)} poses in DB.")
        for r in res[:2]:
            print(r)
        
        # Write to a mapping file for fix script
        with open('poses_db.json', 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False)
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code, e.reason)
    print(e.read().decode())
