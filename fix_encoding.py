import sys

filepath = '3dgs_viewer/spark-3dgs-viewer/src/components/SparkGaussianViewer.vue'

with open(filepath, 'rb') as f:
    content = f.read()

print(f'Original size: {len(content)} bytes')

bom = b'\xef\xbb\xbf'
has_bom = content[:3] == bom
print(f'Starts with BOM: {has_bom}')

has_crlf = b'\r\n' in content
print(f'Has CRLF: {has_crlf}')

if has_bom:
    content = content[3:]
    print('Removed BOM')

original_len = len(content)
content = content.replace(b'\r\n', b'\n')
if len(content) != original_len:
    print(f'Converted CRLF->LF ({original_len - len(content)} bytes saved)')

with open(filepath, 'wb') as f:
    f.write(content)

print(f'Final size: {len(content)} bytes')
print('Done')
