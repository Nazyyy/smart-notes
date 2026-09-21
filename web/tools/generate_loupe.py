from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps

ROOT = Path("/home/dima/qq")
OUT = ROOT / "assets" / "loupe"
OUT.mkdir(parents=True, exist_ok=True)
FONT_DIR = ROOT / "assets" / "fonts"
PANGOLIN = str(FONT_DIR / "Pangolin-Regular.ttf")
MARCK = str(FONT_DIR / "MarckScript-Regular.ttf")
DEJAVU = "/usr/share/fonts/TTF/DejaVuSans.ttf"
DEJAVU_B = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"

def fnt(path, size):
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.truetype(DEJAVU, size=size)

W, H = 960, 1200

def render_raw():
    rng = random.Random(42)
    raw = Image.new("RGB", (W, H), (242, 233, 210))
    px = raw.load()
    for _ in range(W * H // 16):
        x, y = rng.randrange(W), rng.randrange(H)
        d = rng.randint(-16, 10)
        r, g, b = px[x, y]
        px[x, y] = (max(0, min(255, r + d)), max(0, min(255, g + d - 1)), max(0, min(255, b + d - 2)))
    draw = ImageDraw.Draw(raw, "RGBA")
    mx = 96
    draw.line([(mx, 0), (mx, H)], fill=(196, 86, 86, 175), width=3)
    y0 = 78
    while y0 < H - 24:
        draw.line([(28, y0), (W - 24, y0)], fill=(90, 130, 180, 140), width=1)
        y0 += 36
    for cy in (70, H // 2, H - 80):
        draw.ellipse((18, cy - 14, 48, cy + 14), fill=(230, 220, 198), outline=(160, 150, 130))
        draw.ellipse((24, cy - 8, 42, cy + 8), fill=(32, 30, 26))
    ink = (32, 44, 82, 230)
    red = (150, 40, 40, 220)
    body = fnt(PANGOLIN, 30)
    head = fnt(MARCK, 46)
    tiny = fnt(PANGOLIN, 20)
    draw.text((118, 22), "Физика · 9 класс", font=tiny, fill=(90, 80, 70, 200))
    draw.text((W - 210, 22), "21.01.2026", font=tiny, fill=(90, 80, 70, 200))
    draw.text((118, 48), "Кинетическая энергия", font=head, fill=ink)
    lines = [
        "Ek — энергия движения тела.",
        "Чем больше масса и скорость,",
        "тем больше запас энергии.",
        "",
        "Формула:",
        "    Ek = m v^2 / 2",
        "",
        "где  m — масса, кг",
        "     v — скорость, м/с",
        "     Ek — джоули, Дж",
        "",
        "Пример: m = 2 кг, v = 3 м/с",
        "Ek = 2 * 9 / 2 = 9 Дж",
        "",
        "Не забыть: если v=0, то Ek=0.",
        "Потенциальная — отдельно (Ep=mgh).",
    ]
    y = 118
    for line in lines:
        col = red if "Не забыть" in line else ink
        x = 122
        for ch in line:
            jx = rng.uniform(-0.25, 0.3)
            jy = rng.uniform(-0.5, 0.45)
            rot = rng.uniform(-0.9, 0.9)
            bbox = body.getbbox(ch)
            cw = max(1, bbox[2] - bbox[0] + 8)
            chh = max(1, bbox[3] - bbox[1] + 10)
            tile = Image.new("RGBA", (cw + 12, chh + 12), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((4, 2), ch, font=body, fill=col)
            tile = tile.rotate(rot, resample=Image.BICUBIC, expand=True)
            raw.paste(tile, (int(x + jx), int(y + jy)), tile)
            x += (bbox[2] - bbox[0]) + rng.uniform(0.3, 1.2)
        y += 36
    raw = ImageEnhance.Contrast(raw.convert("RGB")).enhance(1.06)
    raw = raw.filter(ImageFilter.UnsharpMask(radius=1.0, percent=70, threshold=3))
    noise = Image.effect_noise(raw.size, 10).convert("L")
    raw = Image.blend(raw, ImageOps.colorize(noise, (18, 16, 12), (250, 244, 228)), 0.06)
    raw = raw.rotate(-0.28, resample=Image.BICUBIC, fillcolor=(28, 26, 24))
    raw.save(OUT / "raw.jpg", "JPEG", quality=84, optimize=True)

def render_clean():
    clean = Image.new("RGB", (W, H), (248, 248, 246))
    cd = ImageDraw.Draw(clean)
    for yy in range(H):
        shade = 248 - int(6 * yy / H)
        cd.line([(0, yy), (W, yy)], fill=(shade, shade, shade - 1))
    cd.rectangle((0, 0, 4, H), fill=(230, 230, 230))
    title = fnt(DEJAVU_B, 28)
    ui = fnt(DEJAVU, 20)
    sm = fnt(DEJAVU, 16)
    mono = fnt(DEJAVU, 22)
    cd.text((48, 36), "GLYPH  ·  VECTOR OUTPUT", font=sm, fill=(120, 120, 120))
    cd.text((48, 68), "Кинетическая энергия", font=title, fill=(18, 18, 18))
    cd.text((48, 108), "Физика · 9 класс  ·  21.01.2026", font=sm, fill=(130, 130, 130))
    cd.line((48, 140, W - 48, 140), fill=(220, 220, 220), width=1)
    y = 168
    for line in [
        "Ek — энергия движения тела.",
        "Чем больше масса и скорость, тем больше запас энергии.",
        "",
        "Определение",
    ]:
        cd.text((48, y), line, font=ui, fill=(28, 28, 28))
        y += 32
    cd.rounded_rectangle((48, 310, W - 48, 470), radius=14, fill=(18, 18, 18))
    eq = fnt(DEJAVU, 42)
    cd.text((72, 338), "Ek  =  m v^2  /  2", font=eq, fill=(245, 245, 245))
    cd.text((72, 410), "SI  ·  джоуль (Дж)  ·  кг · м²/с²", font=sm, fill=(170, 170, 170))
    y = 500
    for k, desc in [("m", "масса тела, кг"), ("v", "скорость, м/с"), ("Ek", "кинетическая энергия, Дж")]:
        cd.ellipse((52, y + 4, 64, y + 16), fill=(28, 28, 28))
        cd.text((78, y), k + "  —  " + desc, font=ui, fill=(40, 40, 40))
        y += 36
    cd.rounded_rectangle((48, 640, W - 48, 760), radius=12, outline=(210, 210, 210), width=1)
    cd.text((68, 660), "Пример", font=sm, fill=(120, 120, 120))
    cd.text((68, 690), "m = 2 кг,   v = 3 м/с", font=ui, fill=(20, 20, 20))
    cd.text((68, 722), "Ek = 2 · 9 / 2 = 9 Дж", font=mono, fill=(20, 20, 20))
    cd.text((48, 800), "Если v = 0, то Ek = 0. Потенциальная энергия:", font=ui, fill=(40, 40, 40))
    cd.text((48, 838), "Ep = m g h", font=mono, fill=(20, 20, 20))
    cd.line((48, H - 90, W - 48, H - 90), fill=(220, 220, 220), width=1)
    cd.text((48, H - 70), "RULER_GRID_STRIPPED  100%", font=sm, fill=(110, 110, 110))
    cd.text((W - 260, H - 70), "4.8 MB  ->  86 KB", font=sm, fill=(110, 110, 110))
    clean.save(OUT / "clean.jpg", "JPEG", quality=90, optimize=True)

if __name__ == "__main__":
    render_raw()
    render_clean()
    print("raw", (OUT / "raw.jpg").stat().st_size)
    print("clean", (OUT / "clean.jpg").stat().st_size)
