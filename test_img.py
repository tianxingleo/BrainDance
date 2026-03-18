import urllib.request
import urllib.error

base = 'https://supabase.tianxingleo.top/storage/v1/object/public/braindance-assets/test1/scene_party_001/output/images/frame_'

found = []
for i in range(1, 91):
    num = str(i).zfill(5)
    url = base + num + '.jpg'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        urllib.request.urlopen(req)
        print(url, "EXISTS")
        found.append(num)
    except urllib.error.HTTPError as e:
        pass
print("Found images:", found)
