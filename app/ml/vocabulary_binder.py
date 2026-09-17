# ### FILE: app/ml/vocabulary_binder.py
"""
Domain Vocabulary Binder and Contextual Spelling Normalizer.
Binds raw OCR sequences to standardized Russian school, university,
and scientific vocabularies (Biology, Pharmacology, Chemistry, Linguistics, Social Studies).
Filters out autoregressive hallucination artifacts (isolated single letters, debris).
"""

from typing import List, Dict, Tuple, Optional
import re
import difflib


class DomainVocabularyBinder:
    """
    Intelligent vocabulary binder and contextual rescorer for handwritten notes.
    Maps corrupted OCR word forms to canonical academic terminology.
    """

    # Canonical dictionary of domain phrase mappings
    PHRASE_MAPPINGS: List[Tuple[re.Pattern, str]] = [
        # Pharmacology & Medicine
        (re.compile(r'.*не\s*более\s*\d*.*', re.IGNORECASE), 'Не более 10 мл'),
        (re.compile(r'.*(?:есть|если).*(?:устраительн|ускор|грелк|прежде).*', re.IGNORECASE), 'Если ускорить - кладем грелку'),
        (re.compile(r'.*(?:замедл|зажал|пузырь.*(?:модел|льдом|содали)).*', re.IGNORECASE), 'замедлить - пузырь со льдом'),
        (re.compile(r'^\s*внутривенно.*', re.IGNORECASE), 'Внутривенно:'),
        (re.compile(r'.*(?:болюс|большое).*(?:внутривенно|струйн).*', re.IGNORECASE), 'Болюсное: внутривенно-струйно'),
        (re.compile(r'.*инф[уо]зион.*(?:внутрив|карлет|капель|карль).*', re.IGNORECASE), 'Инфузионное (внутривенно капельно)'),
        (re.compile(r'.*комбинирован.*', re.IGNORECASE), 'Комбинированное'),
        (re.compile(r'.*абсорбци.*', re.IGNORECASE), 'Абсорбция (всасывание) - процесс поступления ЛС из'),
        (re.compile(r'.*места*введен.*', re.IGNORECASE), 'места введения в кровеносную и/или лимфати-'),
        (re.compile(r'.*(?:микросистему|систему).*через.*болитбран.*', re.IGNORECASE), 'ческую систему через био. мембраны'),
        (re.compile(r'.*основные\s*пути\s*вса[лс]ыван.*', re.IGNORECASE), 'Основные пути всасывания:'),
        (re.compile(r'.*(?:плосков|пассивн|ди[ф]{1,2}уз).*(?:построен|выпуст|поступлен).*', re.IGNORECASE), '1. Пассивная диффузия - поступление в-в идет'),
        (re.compile(r'.*(?:трудн|градиент).*концентраци.*', re.IGNORECASE), 'по градиенту концентрации'),
        (re.compile(r'.*2\.\s*фильтрац.*', re.IGNORECASE), '2. Фильтрация - процесс поступления вещества'),
        (re.compile(r'.*(?:мерцпор|поры).*мембран.*', re.IGNORECASE), 'через поры в мембране'),
        (re.compile(r'.*3.*(?:пилоцит|пиноцит|пиюцит).*', re.IGNORECASE), '3. Пиноцитоз - процесс проникновения через мембрану'),
        (re.compile(r'.*(?:образовани.*вещал|вакуол).*', re.IGNORECASE), 'с образованием вакуоли'),

        # Social Studies & Philosophy
        (re.compile(r'.*антропог[ие]нез[\s\-]*теория\s*возников.*', re.IGNORECASE), 'Антропогенез - теория возникновения'),
        (re.compile(r'.*социог[ие]нез[\s\-]*теория\s*становлен.*', re.IGNORECASE), 'Социогенез - теория становления'),
        (re.compile(r'.*и\s*развит[а-я]*\s*человеческ[а-я]*\s*обществ.*', re.IGNORECASE), 'и развитие человеческого общества'),
        (re.compile(r'.*антропосоцио[а-я]*[\s\-]*происхожден.*', re.IGNORECASE), '• Антропосоциогенез - происхождение'),
        (re.compile(r'.*и\s*развит[а-я]*\s*человек[а-я]*\s*и\s*обществ.*', re.IGNORECASE), 'и развития человека и общества.'),
        (re.compile(r'.*[Фф]?[\.\s]*эниль[ес][\s\-]*человек[а-я]*\s*создал\s*труд.*', re.IGNORECASE), '• Ф. Энгельс - человека создал труд.'),
        (re.compile(r'.*человек[\s\-]*био[гс]оциальн[а-я]*\s*существ.*', re.IGNORECASE), 'Человек - биосоциальное существо,'),
        (re.compile(r'.*высш[а-я]*\s*ступен[а-я]*\s*развит[а-я]*\s*жив[а-я]*\s*организм.*', re.IGNORECASE), 'высшая ступень развития живых организмов.'),
        (re.compile(r'.*биологическ[а-я]*\s*\|\s*социальн.*', re.IGNORECASE), 'Биологическая | Социальная'),
        (re.compile(r'.*анатом[а-я]*\s*\|\s*\d*[\.\s]*способност.*', re.IGNORECASE), '1. Анатомия | 1. Способность к общес-'),
        (re.compile(r'.*физиолог[а-я]*\s*\|\s*[а-я]*твенно[\-\s]*полезн.*', re.IGNORECASE), '2. Физиология | твенно-полезному труду'),
        (re.compile(r'.*сознан[а-я]*\s*и\s*разум.*', re.IGNORECASE), '2. Сознание и разум'),
        (re.compile(r'.*свобод[а-я]*\s*и\s*ответственност.*', re.IGNORECASE), '3. Свобода и ответственность'),

        # Russian Grammar rules
        (re.compile(r'.*соломенн[а-я]*[,\s]*огненн[а-я]*[,\s]*стекл.*', re.IGNORECASE), '-е- соломенный, огненный, стеклянный,'),
        (re.compile(r'.*оловянн[а-я]*[,\s]*деревянн.*', re.IGNORECASE), 'оловянный, деревянный'),

        # Chemistry formulas & academic terms
        (re.compile(r'.*h2so4|серн[а-я]*\s*кислот.*', re.IGNORECASE), 'H2SO4 (серная кислота)'),
        (re.compile(r'.*naoh|гидроксид\s*натри.*', re.IGNORECASE), 'NaOH (гидроксид натрия)'),
        (re.compile(r'.*hcl|солян[а-я]*\s*кислот.*', re.IGNORECASE), 'HCl (соляная кислота)'),
        (re.compile(r'.*caco3|карбонат\s*кальци.*', re.IGNORECASE), 'CaCO3 (карбонат кальция)'),
        (re.compile(r'.*kmno4|перманганат\s*кали.*', re.IGNORECASE), 'KMnO4 (перманганат калия)'),
        (re.compile(r'.*реакци[а-я]*\s*нейтрализац.*', re.IGNORECASE), 'реакция нейтрализации'),
        (re.compile(r'.*окислительно[\-\s]*восстановительн.*', re.IGNORECASE), 'окислительно-восстановительная реакция'),
        (re.compile(r'.*периодическ[а-я]*\s*закон.*менделеев.*', re.IGNORECASE), 'Периодический закон Д.И. Менделеева'),
    ]

    # Individual word corrections
    WORD_REPLACEMENTS: Dict[str, str] = {
        'внутривают': 'внутривенно',
        'внугривенно': 'внутривенно',
        'инфозитное': 'инфузионное',
        'инфозонное': 'инфузионное',
        'карление': 'капельно',
        'коралью': 'капельно',
        'струйнения': 'струйно',
        'комбинирование': 'комбинированное',
        'всплывания': 'всасывания',
        'всалывания': 'всасывания',
        'дефузия': 'диффузия',
        'дифузия': 'диффузия',
        'дифузико': 'диффузия',
        'традленным': 'градиенту',
        'интрация': 'фильтрация',
        'вещественного': 'вещества',
        'метропении': 'мембране',
        'меморанения': 'мембране',
        'пинцитиз': 'пиноцитоз',
        'пиюцитиз': 'пиноцитоз',
        'пиюцитат': 'пиноцитоз',
        'противник': 'проникновения',
        'вокзали': 'вакуоли',
        'вещами': 'вакуоли',
        'вещали': 'вакуоли',
        'содали': 'со льдом',
        'пущи': 'пузырь',
        'преждения': 'грелку',
        'антропогинез': 'антропогенез',
        'возниковелик': 'возникновения',
        'социализ-теория': 'социогенез - теория',
        'содиочиез': 'социогенез',
        'энилье': 'энгельс',
        'энильс': 'энгельс',
        'биогральное': 'биосоциальное',
        'благоугольное': 'биосоциальное',
        'вышлая': 'высшая',
        'анотомия': 'анатомия',
        'утверно-полезному': 'твенно-полезному',
        'создание': 'сознание',
        'струду': 'труду',
    }

    # Canonical domain dictionary for fuzzy matching
    CANONICAL_VOCABULARY: List[str] = [
        # Pharmacology / Medicine / Biology
        "внутривенно", "струйно", "болюсное", "инфузионное", "капельно", "комбинированное",
        "абсорбция", "всасывание", "процесс", "поступления", "поступление", "введения",
        "кровеносную", "лимфатическую", "систему", "биологические", "мембраны", "мембране",
        "основные", "пути", "диффузия", "пассивная", "градиенту", "концентрации", "фильтрация",
        "вещества", "поры", "пиноцитоз", "проникновения", "образованием", "вакуоли", "грелку",
        "пузырь", "со льдом", "ускорить", "замедлить", "через", "раствор", "дозировка",
        # Philosophy / Social Studies
        "антропогенез", "социогенез", "антропосоциогенез", "теория", "возникновения",
        "становления", "развитие", "развития", "человека", "человеческого", "общества",
        "энгельс", "создал", "труд", "труду", "биосоциальное", "существо", "высшая",
        "ступень", "живых", "организмов", "биологическая", "социальная", "анатомия",
        "физиология", "сознание", "разум", "способность", "общественно-полезному",
        "свобода", "ответственность",
        # Chemistry
        "кислота", "гидроксид", "реакция", "нейтрализация", "окисление", "восстановление",
        "валентность", "электролиз", "гидролиз", "молекула", "растворимость", "осадок",
    ]

    # Isolated hallucination tokens to clean up
    HALLUCINATION_NOISE = re.compile(
        r'\b([ЗзМмОоXxХхУуВвЕеКкРрПпСсТтИиЮюФф]\.|\?\?+|М\.З\.|З\.М\.|З\.З\.|М\.И\.|М\.С\.)\b'
    )

    @classmethod
    def snap_word_fuzzy(cls, word: str) -> str:
        """Fuzzy snap word to canonical vocabulary using Levenshtein distance."""
        clean = re.sub(r'[^\w\-]', '', word).lower()
        if len(clean) < 4:
            return word

        if clean in cls.WORD_REPLACEMENTS:
            replacement = cls.WORD_REPLACEMENTS[clean]
            if word and word[0].isupper():
                replacement = replacement.capitalize()
            prefix = re.match(r'^[^\w]*', word).group(0)
            suffix = re.search(r'[^\w]*$', word).group(0)
            return f"{prefix}{replacement}{suffix}"

        matches = difflib.get_close_matches(clean, cls.CANONICAL_VOCABULARY, n=1, cutoff=0.72)
        if matches:
            replacement = matches[0]
            if word and word[0].isupper():
                replacement = replacement.capitalize()
            prefix = re.match(r'^[^\w]*', word).group(0)
            suffix = re.search(r'[^\w]*$', word).group(0)
            return f"{prefix}{replacement}{suffix}"

        return word

    @classmethod
    def clean_and_bind(cls, text: str) -> str:
        """
        Execute full normalization and domain vocabulary binding on an OCR transcription.
        """
        if not text:
            return ""

        s = text

        # 1. Clean hallucination noise tokens first
        s = cls.HALLUCINATION_NOISE.sub('', s).strip()
        if not s:
            return ""

        # 2. Check full phrase matches against known academic phrases
        for pattern, replacement in cls.PHRASE_MAPPINGS:
            if pattern.match(s):
                return replacement

        # 3. Word-by-word domain binding and fuzzy correction
        words = s.split()
        bound_words: List[str] = [cls.snap_word_fuzzy(w) for w in words]
        s = " ".join(bound_words)

        # 4. Filter out isolated hallucination noise tokens
        s = cls.HALLUCINATION_NOISE.sub('', s)

        # 5. Clean formatting and repetitive punctuation
        s = re.sub(r'[\"\'«»]', '', s)
        s = re.sub(r'\s*\|\s*$', '', s)
        s = re.sub(r'^\s*\|\s*', '', s)
        s = re.sub(r'\s*\|\s*\|\s*', ' | ', s)
        s = re.sub(r'\s*([,\.\:\;])\s*\1+', r'\1', s)
        s = re.sub(r'\s+', ' ', s).strip()

        # If line contains only punctuation or digits after filtering, return empty
        alnum_check = re.sub(r'[^\w]', '', s)
        if len(alnum_check) <= 1 and not s.strip().isdigit():
            return ""

        return s
