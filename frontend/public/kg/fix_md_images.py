import hashlib
import os
import re
import requests

MD_PATH = 'frontend/public/kg/01_知识图谱_架构与功能设计.md'
IMG_DIR = 'frontend/public/kg/images'

os.makedirs(IMG_DIR, exist_ok=True)

with open(MD_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

def url2name(url):
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = os.path.splitext(url.split('?')[0])[1] or '.img'
    return f'{h}{ext}'

pattern = r'!\[([^\]]*)\]\((https?://[^)]+)\)'
matches = list(re.finditer(pattern, content))
url2local = {}

for m in matches:
    url = m.group(2)
    fname = url2name(url)
    fpath = os.path.join(IMG_DIR, fname)
    url2local[url] = f'images/{fname}'
    if not os.path.exists(fpath):
        try:
            print(f'Downloading: {url} -> {fpath}')
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(fpath, 'wb') as imgf:
                imgf.write(r.content)
        except Exception as e:
            print(f'Failed to download {url}: {e}')

def repl(m):
    alt, url = m.group(1), m.group(2)
    return f'![{alt}]({url2local.get(url, url)})'

new_content = re.sub(pattern, repl, content)

with open(MD_PATH, 'w', encoding='utf-8') as f:
    f.write(new_content)

print('图片下载和替换完成！') 