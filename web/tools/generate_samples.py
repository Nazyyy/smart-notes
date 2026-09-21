from pathlib import Path
import json, math, os, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps

ROOT = Path("/home/dima/qq")
OUT = ROOT / "assets" / "samples"
FONT_DIR = ROOT / "assets" / "fonts"
OUT.mkdir(parents=True, exist_ok=True)

PANGOLIN = str(FONT_DIR / "Pangolin-Regular.ttf")
CAVEAT = str(FONT_DIR / "Caveat.ttf")
MARCK = str(FONT_DIR / "MarckScript-Regular.ttf")
DEJAVU = "/usr/share/fonts/TTF/DejaVuSans.ttf"

def font(path, size):
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.truetype(DEJAVU, size=size)

NOTES = [
    {
        "id": "istoriya-1812",
        "title": "История · 10 класс",
        "file": "istoriya-1812.png",
        "subject": "Отечественная война 1812 г.",
        "date": "12.03.2026",
        "body": [
            "Тема: Отечественная война 1812 года",
            "Причины: континентальная блокада Англии,",
            "отказ России её соблюдать, стремление",
            "Наполеона к гегемонии в Европе.",
            "",
            "24 июня — переход Немана. Великая армия",
            "~600 тыс. чел. План Барклая: отступление",
            "и сохранение армии (не дать ген. сражения).",
            "",
            "Смоленское сражение 16–18 авг. — город сожжён.",
            "26 авг. Бородино. Кутузов: «главное — армия».",
            "Потери огромные с обеих сторон, Москва сдана.",
            "",
            "Тарутинский манёвр. Партизаны: Давыдов,",
            "Сеславин, Фигнер. голод + мороз + казаки.",
            "Березина — катастрофа для французов.",
            "",
            "Итог: изгнание, начало заграничных походов",
            "1813–14. Рост национального самосознания.",
        ],
    },
    {
        "id": "biologiya-kletka",
        "title": "Биология · 9 класс",
        "file": "biologiya-kletka.png",
        "subject": "Клетка — единица жизни",
        "date": "04.02.2026",
        "body": [
            "Клетка — структурная и функциональная",
            "единица всего живого. Цитология.",
            "",
            "Прокариоты: нет ядра (бактерии, археи).",
            "Эукариоты: ядро + органоиды.",
            "",
            "Органоиды:",
            "• митохондрии — АТФ, двойная мембрана",
            "• рибосомы — синтез белка (есть у всех)",
            "• ЭПС шероховатая / гладкая",
            "• аппарат Гольджи — упаковка",
            "• лизосомы — «желудок» клетки",
            "• хлоропласты — только растения, фотосинтез",
            "",
            "Мембрана: жидкостно-мозаичная модель",
            "Сингер и Николсон, 1972.",
            "Диффузия, осмос, активный транспорт.",
        ],
    },
    {
        "id": "fizika-newton",
        "title": "Физика · 9 класс",
        "file": "fizika-newton.png",
        "subject": "Законы Ньютона",
        "date": "21.01.2026",
        "body": [
            "I закон (инерция): если F = 0, то v = const",
            "тело покоится или движется равномерно",
            "прямолинейно. ИСО.",
            "",
            "II закон: F = ma   (главный рабочий)",
            "a = F / m     единица силы — ньютон",
            "1 Н = 1 кг·м/с²",
            "",
            "III закон: F1 = −F2   силы действия",
            "и противодействия равны, противоположны,",
            "приложены к РАЗНЫМ телам.",
            "",
            "Вес P = mg (на опору / подвес).",
            "Невесомость: N = 0 при свободном падении.",
            "Трение: Fтр = μN   покой / скольжение.",
            "",
            "Задача: m = 2 кг, a = 3 м/с² → F = 6 Н",
        ],
    },
    {
        "id": "literatura-onegin",
        "title": "Литература · 9 класс",
        "file": "literatura-onegin.png",
        "subject": "Евгений Онегин",
        "date": "18.04.2026",
        "body": [
            "Роман в стихах, «энциклопедия русской жизни»",
            "(Белинский). Онегинская строфа: 14 строк,",
            "ямб, схема AbAb CCdd EffE gg.",
            "",
            "Онегин — «лишний человек»: умён, холоден,",
            "скучает, не умеет любить. Дуэль с Ленским —",
            "страх света, а не честь.",
            "",
            "Татьяна: «русская душою», письмо — искренность.",
            "В финале она уже светская дама, но любит",
            "по-прежнему. «Я другому отдана».",
            "",
            "Тема: судьба, долг, несбывшаяся любовь.",
            "Автор присутствует как герой-рассказчик.",
            "Лирические отступления — про театр, балы,",
            "ножки, деревню и свободу.",
        ],
    },
    {
        "id": "himiya-rastvory",
        "title": "Химия · 8 класс",
        "file": "himiya-rastvory.png",
        "subject": "Растворы и концентрации",
        "date": "09.12.2025",
        "body": [
            "Раствор = растворитель + растворённое в-во.",
            "Массовая доля:  w = m(в-ва) / m(р-ра)",
            "w · 100% = процентная концентрация.",
            "",
            "Пример: 20 г соли в 180 г воды.",
            "m(р-ра) = 200 г   w = 20/200 = 0,1 = 10%",
            "",
            "Молярность: C = ν / V   [моль/л]",
            "ν = m / M",
            "",
            "Растворимость зависит от T. Насыщенный /",
            "ненасыщенный / пересыщенный.",
            "Кристаллогидраты: CuSO4 · 5H2O — медный купорос.",
            "",
            "Не забыть: дома задача №14, стр. 87.",
            "контрольная в пятницу!!!",
        ],
    },
]

def ruled_paper(w, h, seed):
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), (244, 236, 214))
    px = img.load()
    for _ in range(w * h // 18):
        x, y = rng.randrange(w), rng.randrange(h)
        d = rng.randint(-18, 12)
        r, g, b = px[x, y]
        px[x, y] = (
            max(0, min(255, r + d)),
            max(0, min(255, g + d - 1)),
            max(0, min(255, b + d - 3)),
        )
    draw = ImageDraw.Draw(img, "RGBA")
    margin_x = 118
    draw.line([(margin_x, 0), (margin_x, h)], fill=(196, 86, 86, 170), width=3)
    draw.line([(margin_x + 5, 0), (margin_x + 5, h)], fill=(196, 86, 86, 70), width=1)
    y0 = 96
    gap = 42
    while y0 < h - 30:
        shade = 150 + rng.randint(-8, 8)
        draw.line([(40, y0), (w - 36, y0)], fill=(90, 130, 180, shade), width=1)
        y0 += gap
    for cy in (70, h // 2, h - 80):
        draw.ellipse((28, cy - 16, 60, cy + 16), fill=(232, 224, 200, 255), outline=(170, 160, 140, 200))
        draw.ellipse((34, cy - 10, 54, cy + 10), fill=(40, 38, 34, 255))
    if seed % 2 == 0:
        cx, cy = rng.randint(w - 280, w - 120), rng.randint(80, 220)
        for rad in range(48, 62):
            draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), outline=(150, 100, 60, 18), width=2)
    draw.polygon([(w - 70, 0), (w, 0), (w, 70)], fill=(210, 198, 170, 90))
    return img

def wrap_jitter_line(base, text, xy, fnt, fill, rng):
    x, y = xy
    for ch in text:
        jx = rng.uniform(-0.2, 0.25)
        jy = rng.uniform(-0.45, 0.4)
        rot = rng.uniform(-0.8, 0.8)
        bbox = fnt.getbbox(ch)
        cw = max(1, bbox[2] - bbox[0] + 8)
        chh = max(1, bbox[3] - bbox[1] + 10)
        tile = Image.new("RGBA", (cw + 12, chh + 12), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        td.text((4, 2), ch, font=fnt, fill=fill)
        tile = tile.rotate(rot, resample=Image.BICUBIC, expand=True)
        base.paste(tile, (int(x + jx), int(y + jy - 4)), tile)
        x += (bbox[2] - bbox[0]) + rng.uniform(0.4, 1.6)
    return x

def render_note(spec, seed=1):
    rng = random.Random(seed)
    w, h = 1240, 1650
    paper = ruled_paper(w, h, seed)
    img = paper.convert("RGBA")
    title_f = font(MARCK, 54)
    head_f = font(PANGOLIN, 38)
    body_f = font(PANGOLIN, 34)
    tiny_f = font(PANGOLIN, 22)
    ink = (28, 42, 78, 230)
    ink2 = (40, 38, 70, 210)
    red = (150, 40, 40, 210)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((150, 28), spec["title"], font=tiny_f, fill=(90, 80, 70, 200))
    draw.text((w - 280, 28), spec["date"], font=tiny_f, fill=(90, 80, 70, 200))
    draw.text((150, 58), spec["subject"], font=title_f, fill=ink)
    y = 128
    gap = 42
    for i, line in enumerate(spec["body"]):
        if not line.strip():
            y += gap
            continue
        fnt = head_f if i == 0 else body_f
        col = red if ("контрольная" in line.lower() or "не забыть" in line.lower()) else ink2
        wrap_jitter_line(img, line, (148, y + rng.uniform(-2, 2)), fnt, col, rng)
        if rng.random() < 0.08:
            draw.line((148, y + 36, 900, y + 38), fill=(40, 60, 120, 80), width=1)
        y += gap
    sx, sy = 160, h - 90
    pts = []
    for k in range(18):
        pts.append((sx + k * 14 + rng.randint(-3, 3), sy + math.sin(k / 2) * 10 + rng.randint(-4, 4)))
    draw.line(pts, fill=(30, 40, 70, 160), width=2)
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
    rgb = ImageEnhance.Color(rgb).enhance(0.92)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=3))
    arr_noise = Image.effect_noise(rgb.size, 12).convert("L")
    rgb = Image.blend(rgb, ImageOps.colorize(arr_noise, (20, 18, 14), (250, 246, 230)), 0.07)
    angle = rng.uniform(-0.35, 0.4)
    rgb = rgb.rotate(angle, resample=Image.BICUBIC, fillcolor=(30, 28, 26))
    vig = Image.new("L", rgb.size, 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse((-80, -80, rgb.size[0] + 80, rgb.size[1] + 80), fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(42))
    rgb = Image.composite(rgb, ImageEnhance.Brightness(rgb).enhance(0.72), vig)
    path = OUT / spec["file"]
    rgb.save(path, "PNG", optimize=True)
    thumb = rgb.copy()
    thumb.thumbnail((420, 560))
    thumb.save(OUT / spec["file"].replace(".png", "-thumb.jpg"), "JPEG", quality=82)
    return {
        "id": spec["id"],
        "title": spec["title"],
        "subject": spec["subject"],
        "date": spec["date"],
        "file": "assets/samples/" + spec["file"],
        "thumb": "assets/samples/" + spec["file"].replace(".png", "-thumb.jpg"),
        "truth": "\n".join(spec["body"]).strip(),
    }

def main():
    catalog = []
    for i, spec in enumerate(NOTES):
        meta = render_note(spec, seed=11 + i * 17)
        catalog.append(meta)
        print("wrote", meta["file"], os.path.getsize(ROOT / meta["file"]))
    (OUT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print("catalog", len(catalog))

if __name__ == "__main__":
    main()
