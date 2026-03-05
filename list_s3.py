import urllib.request
import json

url = 'https://supabase.tianxingleo.top/storage/v1/object/list/braindance-assets'
key = 'sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH'
headers = {
    'apikey': key,
    'Authorization': f'Bearer {key}',
    'Content-Type': 'application/json'
}

def list_dir(prefix):
    data = {"prefix": prefix, "limit": 100, "offset": 0, "sortBy": {"column": "name", "order": "asc"}}
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            print(f"--- Prefix: {prefix} ---")
            for item in res:
                print(item['name'])
    except Exception as e:
        print(f"Error listing {prefix}: {e}")

list_dir('test1/scene_party_001/')
list_dir('test1/scene_party_001/output/')
list_dir('test1/scene_party_001/output/images/')
