# ### FILE: app/ml/synthetic_generator.py
"""
Synthetic Cyrillic Handwriting Line Generator (Infinite HTR Factory).
Renders realistic academic handwritten lines using complex Google Fonts (Caveat, Bad Script,
Marck Script, Neucha, etc.) on authentic notebook paper (grid, ruled, slant) with ink degradation,
baseline waviness, and natural noise.
"""

from typing import List, Tuple, Optional, Union
from pathlib import Path
import random
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from app.core.logging import get_logger

logger = get_logger(__name__)


# Academic, scientific, and handwritten Russian training corpus
ACADEMIC_CORPUS_TEMPLATES: List[str] = [
    # Mathematics & Analysis
    "Функция f(x) непрерывна на отрезке [a, b]",
    "Предел lim (sin x)/x = 1 при x -> 0",
    "Производная функции y = x^3 - 3x^2 + 2",
    "Интеграл от 0 до pi sin(x) dx = 2",
    "Теорема Пифагора: a^2 + b^2 = c^2",
    "Дискриминант D = b^2 - 4ac",
    "Векторное произведение [a, b] = -[b, a]",
    "Матрица Грама и определитель det(A) != 0",
    "Ряд Тейлора в окрестности точки x_0",
    "Экспоненциальный закон роста e^(kx)",
    # Physics & Mechanics
    "Второй закон Ньютона: F = m * a",
    "Закон всемирного тяготения F = G*m1*m2/r^2",
    "Кинетическая энергия E_k = m * v^2 / 2",
    "Потенциал электростатического поля фи = q / (4*pi*eps*r)",
    "Уравнение Менделеева-Клапейрона: P*V = nu*R*T",
    "Закон Ома для участка цепи: I = U / R",
    "Сила Лоренца F = q * [v, B]",
    "Период колебаний пружинного маятника T = 2*pi*sqrt(m/k)",
    "Первый закон термодинамики: Q = Delta U + A",
    # Chemistry & Biology
    "Реакция нейтрализации: HCl + NaOH -> NaCl + H2O",
    "Концентрированная серная кислота H2SO4",
    "Окислительно-восстановительная реакция KMnO4",
    "Синтез аммиака: N2 + 3H2 <=> 2NH3 + Q",
    "Органическое соединение CH3-COOH",
    "Строение клетки: митохондрии, рибосомы, ядро",
    "Двуспиральная структура молекулы ДНК",
    "Фотосинтез: 6CO2 + 6H2O -> C6H12O6 + 6O2",
    "Абсорбция лекарственных веществ через биомембраны",
    "Ферментативный катализ и кинетика Михаэлиса-Ментен",
    # Linguistics & Grammar
    "СП с бессоюзной и подчинительной связью",
    "Причастный оборот, выделяемый запятыми",
    "Обособленное обстоятельство времени и причины",
    "Сложноподчинённое предложение с придаточным цели",
    "Правописание гласных в корнях с чередованием",
    "Безударные окончания существительных 1 склонения",
    "Двоеточие в бессоюзном сложном предложении",
    # History & Philosophy
    "Антропосоциогенез: происхождение человека и общества",
    "Биологические и социальные факторы развития",
    "Эпоха Просвещения и теория общественного договора",
    "Экономический базис и идеологическая надстройка",
    "Разделение властей: законодательная, исполнительная, судебная",
]


class SyntheticHandwritingGenerator:
    """
    Renders realistic synthetic handwriting text crops using varied fonts, paper textures,
    ink physics, and baseline deformations.
    """

    def __init__(self, fonts_dir: Optional[Union[str, Path]] = None) -> None:
        if fonts_dir is None:
            self.fonts_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fonts"
        else:
            self.fonts_dir = Path(fonts_dir)

        self.font_paths = list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf"))
        if not self.font_paths:
            logger.warning("No font files found in %s. Synthetic generator will need fonts.", self.fonts_dir)
        else:
            logger.info("SyntheticHandwritingGenerator loaded %d fonts from %s", len(self.font_paths), self.fonts_dir)

    def _draw_paper_background(
        self,
        width: int,
        height: int,
        paper_type: str = "random",
    ) -> np.ndarray:
        """
        Create realistic notebook paper backgrounds:
        - blank (with slight paper grain)
        - grid (quad/клетка 5x5mm)
        - ruled (линейка)
        - slant_ruled (косая линия)
        """
        if paper_type == "random":
            paper_type = random.choice(["blank", "grid", "ruled", "slant_ruled", "blank", "grid"])

        # Base paper color: slightly warm off-white
        bg_r = random.randint(245, 255)
        bg_g = random.randint(243, 253)
        bg_b = random.randint(235, 248)
        img = np.full((height, width, 3), (bg_b, bg_g, bg_r), dtype=np.uint8)

        # Light paper grain noise
        noise = np.random.normal(0, random.uniform(1.5, 4.0), (height, width, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        grid_color = (
            random.randint(180, 215),  # B
            random.randint(190, 220),  # G
            random.randint(190, 220),  # R
        )

        if paper_type == "grid":
            step = random.randint(20, 28)
            offset_x = random.randint(0, step - 1)
            offset_y = random.randint(0, step - 1)
            # Vertical lines
            for x in range(offset_x, width, step):
                cv2.line(img, (x, 0), (x, height), grid_color, 1, lineType=cv2.LINE_AA)
            # Horizontal lines
            for y in range(offset_y, height, step):
                cv2.line(img, (0, y), (width, y), grid_color, 1, lineType=cv2.LINE_AA)

        elif paper_type == "ruled":
            step = random.randint(24, 34)
            offset_y = random.randint(5, step - 1)
            for y in range(offset_y, height, step):
                cv2.line(img, (0, y), (width, y), grid_color, 1, lineType=cv2.LINE_AA)
            # Optional faint red margin line
            if random.random() < 0.25:
                margin_x = random.randint(15, 35)
                margin_color = (random.randint(160, 190), random.randint(160, 190), random.randint(220, 245))
                cv2.line(img, (margin_x, 0), (margin_x, height), margin_color, 1, lineType=cv2.LINE_AA)

        elif paper_type == "slant_ruled":
            step = random.randint(24, 32)
            for y in range(random.randint(0, 10), height, step):
                cv2.line(img, (0, y), (width, y), grid_color, 1, lineType=cv2.LINE_AA)
            slant_step = random.randint(30, 45)
            for x in range(-height, width + height, slant_step):
                pt1 = (x, 0)
                pt2 = (x + int(height * 0.4), height)
                cv2.line(img, pt1, pt2, grid_color, 1, lineType=cv2.LINE_AA)

        return img

    def _get_random_ink_color(self) -> Tuple[int, int, int]:
        """
        Generate realistic pen ink RGB colors:
        - Classic blue ballpoint
        - Dark gel blue
        - Dark ink / black
        - Pencil graphite grey
        """
        ink_style = random.choice(["blue_ballpoint", "dark_blue", "black_gel", "graphite", "blue_ballpoint"])
        if ink_style == "blue_ballpoint":
            return (random.randint(20, 50), random.randint(40, 80), random.randint(140, 200))
        elif ink_style == "dark_blue":
            return (random.randint(15, 35), random.randint(25, 55), random.randint(90, 140))
        elif ink_style == "black_gel":
            v = random.randint(25, 55)
            return (v, v, v + random.randint(0, 8))
        else:  # graphite
            v = random.randint(65, 95)
            return (v, v, v)

    def generate_line(
        self,
        text: Optional[str] = None,
        font_path: Optional[Union[str, Path]] = None,
        target_height: int = 48,
        min_width: int = 256,
        paper_type: str = "random",
        add_baseline_wave: bool = True,
        messy_handwriting: bool = False,
    ) -> Tuple[np.ndarray, str]:
        """
        Renders a single synthetic line of handwritten text with physical handwriting traits.
        Returns (bgr_image, transcription).
        """
        if text is None:
            text = random.choice(ACADEMIC_CORPUS_TEMPLATES)

        # Select font
        if font_path is None:
            if not self.font_paths:
                raise RuntimeError("No font files available in data/fonts/")
            chosen_font_path = random.choice(self.font_paths)
        else:
            chosen_font_path = Path(font_path)

        font_size = random.randint(24, 38)
        try:
            font = ImageFont.truetype(str(chosen_font_path), font_size)
        except Exception as exc:
            logger.warning("Could not load font %s, falling back to first available: %s", chosen_font_path, exc)
            font = ImageFont.truetype(str(self.font_paths[0]), font_size)

        # Estimate text bbox
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Allocate canvas with comfortable padding
        pad_x = random.randint(20, 45)
        pad_y = random.randint(12, 22)
        canvas_w = max(min_width, text_w + pad_x * 2)
        canvas_h = max(target_height, text_h + pad_y * 2)

        # Generate paper background in OpenCV BGR
        bg_bgr = self._draw_paper_background(canvas_w, canvas_h, paper_type=paper_type)

        # Convert paper to PIL RGB
        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
        pil_canvas = Image.fromarray(bg_rgb)
        ink_color = self._get_random_ink_color()

        # Render text with character-level baseline jitter to mimic uneven handwriting
        text_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)

        curr_x = pad_x
        base_y = pad_y
        wave_freq = random.uniform(0.04, 0.16)
        if messy_handwriting:
            wave_amp = random.uniform(2.8, 6.0)
        else:
            wave_amp = random.uniform(1.0, 3.5) if add_baseline_wave else 0.0

        for idx, char in enumerate(text):
            # Character offset with sine-wave baseline jitter + micro randomness
            jitter_range = (-1.8, 1.8) if messy_handwriting else (-0.8, 0.8)
            jitter_y = int(math.sin(idx * wave_freq) * wave_amp + random.uniform(*jitter_range))
            y_pos = base_y + jitter_y

            # Render character
            text_draw.text((curr_x, y_pos), char, fill=(*ink_color, 255), font=font)

            # Advance X by character width + slight tracking variation
            c_bbox = dummy_draw.textbbox((0, 0), char, font=font)
            char_w = max(c_bbox[2] - c_bbox[0], font_size // 5)
            spacing_jitter = random.randint(-3, 5) if messy_handwriting else random.randint(-1, 2)
            curr_x += char_w + spacing_jitter

        # Alpha composite text layer over paper background
        pil_canvas.paste(text_layer, (0, 0), mask=text_layer)

        # Convert back to OpenCV BGR for physical ink augmentations
        out_bgr = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)

        # 1. Stroke degradation (erosion / dilation to mimic pen pressure)
        stroke_effect = random.random()
        if stroke_effect < 0.25:
            # Thin dry pen stroke: slight erosion on dark pixels
            kernel = np.ones((2, 2), np.uint8)
            out_bgr = cv2.dilate(out_bgr, kernel, iterations=1)  # dilates background (erodes ink)
        elif stroke_effect > 0.80:
            # Thick juicy fountain pen stroke: slight ink bleed
            kernel = np.ones((2, 2), np.uint8)
            out_bgr = cv2.erode(out_bgr, kernel, iterations=1)

        # 2. Ink absorption / feathering (micro blur)
        if random.random() < 0.70:
            blur_sigma = random.uniform(0.3, 0.7)
            out_bgr = cv2.GaussianBlur(out_bgr, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

        # 3. Slight global rotation / perspective slant
        if random.random() < 0.50:
            rot_angle = random.uniform(-3.5, 3.5)
            center = (canvas_w // 2, canvas_h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
            out_bgr = cv2.warpAffine(
                out_bgr,
                rot_matrix,
                (canvas_w, canvas_h),
                borderMode=cv2.BORDER_REPLICATE,
            )

        # 4. Lighting gradient (simulating room light / phone shadow)
        if random.random() < 0.40:
            gradient = np.linspace(
                random.uniform(0.88, 1.0),
                random.uniform(0.95, 1.06),
                canvas_w,
            )
            out_bgr = np.clip(out_bgr * gradient[None, :, None], 0, 255).astype(np.uint8)

        # Resize to standard height while preserving aspect ratio
        h, w = out_bgr.shape[:2]
        new_w = max(32, int(w * (target_height / float(h))))
        final_img = cv2.resize(out_bgr, (new_w, target_height), interpolation=cv2.INTER_AREA)

        return final_img, text
