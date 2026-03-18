import urllib.request
import urllib.error

urls = [
    'https://supabase.tianxingleo.top/storage/v1/object/public/braindance-assets/test1/scene_party_001/output/webgl_poses.json',
]

for u in urls:
    try:
        r = urllib.request.urlopen(u)
        print('OK:', u, r.status)
    except Exception as e:
        print('Err:', u, e)
