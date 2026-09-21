from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import json, re, subprocess, tempfile, os

ROOT = Path('/home/dima/qq')
SAMPLES = ROOT / 'assets' / 'samples'
catalog = json.loads((SAMPLES / 'catalog.json').read_text(encoding='utf-8'))

def preprocess(im: Image.Image) -> Image.Image:
    im = im.convert('RGB')
    g = ImageOps.grayscale(im)
    g = ImageEnhance.Contrast(g).enhance(1.6)
    g = ImageOps.autocontrast(g, cutoff=2)
    # light denoise
    g = g.filter(ImageFilter.MedianFilter(3))
    w, h = g.size
    px = g.load()
    # remove near-horizontal ruled lines
    for y in range(h):
        dark = 0
        row = [px[x, y] for x in range(w)]
        for v in row:
            if v < 170:
                dark += 1
        if dark > w * 0.55:
            # likely a rule: bleach thin-ish rows
            for x in range(w):
                if 90 < px[x, y] < 200:
                    px[x, y] = 245
    # adaptive-ish threshold
    out = Image.new('L', (w, h))
    op = out.load()
    block = 32
    for by in range(0, h, block):
        for bx in range(0, w, block):
            vals = []
            for y in range(by, min(h, by+block)):
                for x in range(bx, min(w, bx+block)):
                    vals.append(px[x, y])
            avg = sum(vals) / max(1, len(vals))
            thr = avg - 18
            for y in range(by, min(h, by+block)):
                for x in range(bx, min(w, bx+block)):
                    op[x, y] = 0 if px[x, y] < thr else 255
    # crop left binder
    out = out.crop((90, 20, w - 20, h - 20))
    return out.convert('RGB')

def norm(s):
    s = s.lower()
    s = re.sub(r'[^а-яa-z0-9ё\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokens(s):
    return [t for t in norm(s).split() if len(t) > 2]

os.environ['TESSDATA_PREFIX'] = str(ROOT / 'assets' / 'tessdata')
results = []
for item in catalog:
    src = ROOT / item['file']
    im = Image.open(src)
    pre = preprocess(im)
    tmp = Path('/tmp') / (item['id'] + '-pre.jpg')
    pre.save(tmp, 'JPEG', quality=92)
    raw = subprocess.check_output(['tesseract', str(src), 'stdout', '-l', 'rus+eng', '--psm', '6'], stderr=subprocess.DEVNULL).decode('utf-8', 'ignore')
    cln = subprocess.check_output(['tesseract', str(tmp), 'stdout', '-l', 'rus+eng', '--psm', '6'], stderr=subprocess.DEVNULL).decode('utf-8', 'ignore')
    truth = set(tokens(item['truth']))
    raw_t = set(tokens(raw))
    cln_t = set(tokens(cln))
    rec_raw = len(truth & raw_t) / max(1, len(truth))
    rec_cln = len(truth & cln_t) / max(1, len(truth))
    results.append({
        'id': item['id'],
        'recall_raw': round(rec_raw, 3),
        'recall_pre': round(rec_cln, 3),
        'raw_preview': raw[:220].replace('\n',' | '),
        'pre_preview': cln[:220].replace('\n',' | '),
    })
    print(item['id'], 'raw', rec_raw, 'pre', rec_cln)
    print('PREVIEW PRE:', cln[:180].replace('\n',' / '))
    print('---')

print(json.dumps(results, ensure_ascii=False, indent=2))
