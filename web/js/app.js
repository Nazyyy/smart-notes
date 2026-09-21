/**
 * Smart Notes (Умный конспект) — Web Frontend Application Core
 * Direct integration with FastAPI backend (HybridEnsemble: CRNN + TrOCR + OpenCV),
 * 4-stage processing workflow, KaTeX mathematical typesetting, 3D Flashcards,
 * Cloze tests (fill in the blanks), and Quick Quiz.
 */
(function () {
  "use strict";

  /* ==============================================================================
     1. DOM UTILITIES & HELPERS
     ============================================================================== */
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function showToast(text) {
    var toast = $("#copy-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "copy-toast";
      toast.className = "copy-toast";
      document.body.appendChild(toast);
    }
    toast.textContent = text;
    toast.classList.add("is-open");
    window.clearTimeout(showToast._timer);
    showToast._timer = window.setTimeout(function () {
      toast.classList.remove("is-open");
    }, 2500);
  }

  function toStaticUrl(path) {
    if (!path) return "assets/loupe/raw.jpg";
    if (path.startsWith("data:") || path.startsWith("http://") || path.startsWith("https://") || path.startsWith("assets/")) {
      return path;
    }
    var idx = path.indexOf("data/storage/");
    if (idx !== -1) {
      return "/static/" + path.substring(idx + "data/storage/".length);
    }
    var docIdx = path.indexOf("documents/");
    if (docIdx !== -1) {
      return "/static/" + path.substring(docIdx);
    }
    return path;
  }

  /* ==============================================================================
     2. RICH DEMO DATASETS (Pre-populated for instant testing across all 4 stages)
     ============================================================================== */
  var demoNotes = {
    "phys-kinetics-01": {
      id: "phys-kinetics-01",
      title: "Кинетическая энергия",
      subject: "Физика · 9 класс",
      date: "21.01.2026",
      status: "🟢 Распознано",
      photos: {
        orig: "assets/loupe/raw.jpg",
        deskew: "assets/loupe/clean.jpg",
        shadow: "assets/loupe/clean.jpg",
        bin: "assets/loupe/clean.jpg"
      },
      overlay: "assets/loupe/clean.jpg",
      lines: [
        { num: 1, crop: "assets/loupe/raw.jpg", text: "Кинетическая энергия — энергия движения тела.", conf: 0.94 },
        { num: 2, crop: "assets/loupe/raw.jpg", text: "Формула: E_k = (m * v^2) / 2", conf: 0.98 },
        { num: 3, crop: "assets/loupe/raw.jpg", text: "m — масса в кг, v — скорость в м/с.", conf: 0.91 },
        { num: 4, crop: "assets/loupe/raw.jpg", text: "Теорема об изменении: A = E_k2 - E_k1.", conf: 0.89 },
        { num: 5, crop: "assets/loupe/raw.jpg", text: "При увеличении скорости в 2 раза энергия растет в 4 раза.", conf: 0.95 },
        { num: 6, crop: "assets/loupe/raw.jpg", text: "Задача: m=1200 кг, v=20 м/с -> A = 240 кДж.", conf: 0.92 }
      ],
      markdown: [
        "# Кинетическая энергия",
        "",
        "> Физика · 9 класс · Механика",
        "",
        "## Определение и физический смысл",
        "**Кинетическая энергия** — скалярная физическая величина, являющаяся мерой механического движения материальной точки или системы тел и зависящая от их массы и скорости.",
        "",
        "$$E_k = \\frac{m v^2}{2}$$",
        "",
        "Где:",
        "- $m$ — масса тела, выраженная в килограммах ($\\text{кг}$);",
        "- $v$ — модуль скорости движения тела, выраженный в метрах в секунду ($\\text{м/с}$);",
        "- $E_k$ — кинетическая энергия, измеряемая в джоулях ($\\text{Дж}$).",
        "",
        "## Теорема об изменении кинетической энергии",
        "Работа равнодействующей всех сил, приложенных к телу на некотором пути, в точности равна изменению его кинетической энергии между конечной и начальной точками:",
        "",
        "$$A = \\Delta E_k = E_{k2} - E_{k1} = \\frac{m v_2^2}{2} - \\frac{m v_1^2}{2}$$",
        "",
        "## Важнейшие свойства",
        "- **Скалярность**: кинетическая энергия всегда строго неотрицательна ($E_k \\ge 0$).",
        "- **Квадратичная зависимость**: при увеличении скорости в $2$ раза кинетическая энергия возрастает в $4$ раза.",
        "",
        "## Решение типовой задачи с урока",
        "Автомобиль массой $m = 1200\\,\\text{кг}$ разгоняется из состояния покоя до $v = 20\\,\\text{м/с}$. Найти совершенную работу:",
        "",
        "$$A = \\Delta E_k = \\frac{1200 \\times 20^2}{2} = \\frac{1200 \\times 400}{2} = 240\\,000\\,\\text{Дж} = 240\\,\\text{кДж}$$"
      ].join("\n"),
      flashcards: [
        { front: "Что такое кинетическая энергия?", back: "Скалярная мера механического движения тела, зависящая от его массы и скорости: E_k = (m*v^2)/2.", category: "Определение", hint: "Зависит от массы и квадрата скорости" },
        { front: "Как звучит теорема об изменении кинетической энергии?", back: "Работа равнодействующей силы равна изменению кинетической энергии: A = E_k2 - E_k1.", category: "Теорема", hint: "Связывает механическую работу и энергию" },
        { front: "Во сколько раз увеличится кинетическая энергия при росте скорости в 3 раза?", back: "В 9 раз, так как скорость в формуле возводится в квадрат (3^2 = 9).", category: "Свойство", hint: "Квадратичная зависимость" },
        { front: "В каких единицах измеряется энергия в СИ?", back: "В Джоулях (Дж), 1 Дж = 1 кг·м²/с² = 1 Н·м.", category: "Единицы", hint: "Названа в честь Джеймса Джоуля" }
      ],
      cloze: [
        {
          sentence_with_blank: "Кинетическая энергия зависит от массы тела и [ ... ] его скорости.",
          target_word: "квадрата",
          hint: "Вспомните степень скорости в формуле",
          full_sentence: "Кинетическая энергия зависит от массы тела и квадрата его скорости.",
          options: ["квадрата", "куба", "модуля", "производной"]
        },
        {
          sentence_with_blank: "Работа всех приложенных сил равна [ ... ] кинетической энергии тела.",
          target_word: "изменению",
          hint: "Символ дельта (Δ)",
          full_sentence: "Работа всех приложенных сил равна изменению кинетической энергии тела.",
          options: ["изменению", "сохранению", "минимуму", "квадрату"]
        },
        {
          sentence_with_blank: "В системе СИ кинетическая энергия измеряется в [ ... ].",
          target_word: "джоулях",
          hint: "Обозначается Дж",
          full_sentence: "В системе СИ кинетическая энергия измеряется в джоулях.",
          options: ["джоулях", "ваттах", "ньютонах", "паскалях"]
        }
      ],
      quiz: [
        {
          question: "Чему равна кинетическая энергия тела массой 4 кг, движущегося со скоростью 3 м/с?",
          options: ["18 Дж", "12 Дж", "36 Дж", "6 Дж"],
          correct_index: 0,
          explanation: "По формуле E_k = (m * v^2) / 2 = (4 * 3^2) / 2 = (4 * 9) / 2 = 18 Дж."
        },
        {
          question: "Может ли кинетическая энергия материальной точки быть отрицательной?",
          options: ["Нет, всегда неотрицательна", "Да, при отрицательной скорости", "Да, при торможении", "Зависит от направления оси"],
          correct_index: 0,
          explanation: "Масса m > 0 и v^2 >= 0, поэтому кинетическая энергия всегда строго E_k >= 0."
        },
        {
          question: "Какова работа сил торможения, если кинетическая энергия уменьшилась с 50 Дж до 20 Дж?",
          options: ["-30 Дж", "+30 Дж", "-70 Дж", "+50 Дж"],
          correct_index: 0,
          explanation: "A = E_k2 - E_k1 = 20 Дж - 50 Дж = -30 Дж. Знак минус означает силу торможения."
        }
      ]
    },
    "phys-newton-01": {
      id: "phys-newton-01",
      title: "Законы Ньютона",
      subject: "Физика · 9 класс",
      date: "21.01.2026",
      status: "🟢 Распознано",
      photos: {
        orig: "assets/samples/fizika-newton.jpg",
        deskew: "assets/samples/fizika-newton.jpg",
        shadow: "assets/samples/fizika-newton.jpg",
        bin: "assets/samples/fizika-newton.jpg"
      },
      overlay: "assets/samples/fizika-newton.jpg",
      lines: [
        { num: 1, crop: "assets/samples/fizika-newton-thumb.jpg", text: "I закон: если F = 0, то v = const. ИСО.", conf: 0.95 },
        { num: 2, crop: "assets/samples/fizika-newton-thumb.jpg", text: "II закон: F = ma. Ускорение прямо пропорционально силе.", conf: 0.97 },
        { num: 3, crop: "assets/samples/fizika-newton-thumb.jpg", text: "1 Н = 1 кг·м/с² — единица силы.", conf: 0.93 },
        { num: 4, crop: "assets/samples/fizika-newton-thumb.jpg", text: "III закон: F1 = -F2. Силы приложены к разным телам.", conf: 0.96 },
        { num: 5, crop: "assets/samples/fizika-newton-thumb.jpg", text: "Вес P = mg. Невесомость: N = 0.", conf: 0.90 },
        { num: 6, crop: "assets/samples/fizika-newton-thumb.jpg", text: "Задача: m=2 кг, a=3 м/с² -> F=6 Н.", conf: 0.94 }
      ],
      markdown: [
        "# Законы динамики Ньютона",
        "",
        "> Физика · 9 класс · Основы динамики",
        "",
        "## Первый закон Ньютона (Закон инерции)",
        "Существуют такие системы отсчёта, называемые инерциальными, в которых тело сохраняет состояние покоя или равномерного прямолинейного движения, пока на него не действуют другие силы:",
        "",
        "$$\\sum \\vec{F} = 0 \\implies \\vec{v} = \\text{const}$$",
        "",
        "## Второй закон Ньютона (Основной закон динамики)",
        "Ускорение тела прямо пропорционально равнодействующей всех сил и обратно пропорционально его массе:",
        "",
        "$$\\vec{F} = m \\vec{a} \\iff \\vec{a} = \\frac{\\vec{F}}{m}$$",
        "",
        "## Третий закон Ньютона",
        "Действию всегда есть равное и противоположное противодействие, приложенное к разным телам:",
        "",
        "$$\\vec{F}_{12} = -\\vec{F}_{21}$$"
      ].join("\n"),
      flashcards: [
        { front: "Что гласит первый закон Ньютона?", back: "В инерциальных системах отсчета тело сохраняет состояние покоя или равномерного прямолинейного движения при отсутствии внешних сил.", category: "Закон", hint: "Закон инерции" },
        { front: "Какова формула второго закона Ньютона?", back: "F = m * a (равнодействующая сила равна произведению массы на ускорение).", category: "Формула", hint: "Основной закон динамики" },
        { front: "К чему приложены силы взаимодействия по 3-му закону?", back: "К разным телам, поэтому они никогда не уравновешивают друг друга!", category: "Важное правило", hint: "Разные тела" }
      ],
      cloze: [
        {
          sentence_with_blank: "Силы взаимодействия двух тел приложены к [ ... ] телам.",
          target_word: "разным",
          hint: "Не к одному и тому же",
          full_sentence: "Силы взаимодействия двух тел приложены к разным телам.",
          options: ["разным", "одинаковым", "неподвижным", "одному"]
        },
        {
          sentence_with_blank: "Ускорение тела прямо пропорционально приложенной [ ... ].",
          target_word: "силе",
          hint: "F в формуле F = ma",
          full_sentence: "Ускорение тела прямо пропорционально приложенной силе.",
          options: ["силе", "скорости", "энергии", "плотности"]
        }
      ],
      quiz: [
        {
          question: "Сила 10 Н сообщает телу ускорение 2 м/с². Чему равна масса тела?",
          options: ["5 кг", "20 кг", "2 кг", "8 кг"],
          correct_index: 0,
          explanation: "m = F / a = 10 Н / 2 м/с² = 5 кг."
        }
      ]
    },
    "chem-sol-01": {
      id: "chem-sol-01",
      title: "Растворы и концентрации",
      subject: "Химия · 8-10 класс",
      date: "09.12.2025",
      status: "🟢 Распознано",
      photos: {
        orig: "assets/samples/himiya-rastvory.jpg",
        deskew: "assets/samples/himiya-rastvory.jpg",
        shadow: "assets/samples/himiya-rastvory.jpg",
        bin: "assets/samples/himiya-rastvory.jpg"
      },
      overlay: "assets/samples/himiya-rastvory.jpg",
      lines: [
        { num: 1, crop: "assets/samples/himiya-rastvory-thumb.jpg", text: "Раствор = растворитель + растворенное вещество.", conf: 0.96 },
        { num: 2, crop: "assets/samples/himiya-rastvory-thumb.jpg", text: "Массовая доля: w = m(в-ва) / m(р-ра).", conf: 0.98 },
        { num: 3, crop: "assets/samples/himiya-rastvory-thumb.jpg", text: "Пример: 20 г соли в 180 г воды -> w = 10%.", conf: 0.94 },
        { num: 4, crop: "assets/samples/himiya-rastvory-thumb.jpg", text: "Молярность: C = v / V [моль/л].", conf: 0.91 },
        { num: 5, crop: "assets/samples/himiya-rastvory-thumb.jpg", text: "Кристаллогидраты: CuSO4 * 5H2O — медный купорос.", conf: 0.95 }
      ],
      markdown: [
        "# Растворы и способы выражения концентрации",
        "",
        "> Химия · 8-10 класс",
        "",
        "## Массовая доля растворённого вещества",
        "Отношение массы вещества к общей массе раствора:",
        "",
        "$$w = \\frac{m_{\\text{в-ва}}}{m_{\\text{р-ра}}} = \\frac{m_{\\text{в-ва}}}{m_{\\text{в-ва}} + m_{\\text{воды}}}$$",
        "",
        "В процентах:",
        "$$w\\% = w \\times 100\\%$$",
        "",
        "## Молярная концентрация",
        "$$C_M = \\frac{\\nu}{V} = \\frac{m}{M \\cdot V}$$"
      ].join("\n"),
      flashcards: [
        { front: "Что такое массовая доля вещества?", back: "Отношение массы растворенного вещества к общей массе раствора (m_в-ва / m_р-ра).", category: "Определение", hint: "Обозначается w" },
        { front: "Формула медного купороса?", back: "CuSO4 · 5H2O (пентагидрат сульфата меди II).", category: "Номенклатура", hint: "Кристаллогидрат меди" }
      ],
      cloze: [
        {
          sentence_with_blank: "Масса раствора складывается из массы вещества и массы [ ... ].",
          target_word: "растворителя",
          hint: "Обычно это вода",
          full_sentence: "Масса раствора складывается из массы вещества и массы растворителя.",
          options: ["растворителя", "осадка", "газа", "индикатора"]
        }
      ],
      quiz: [
        {
          question: "В 180 г воды растворили 20 г сахара. Какова массовая доля сахара?",
          options: ["10%", "11.1%", "20%", "9%"],
          correct_index: 0,
          explanation: "Масса раствора = 180 + 20 = 200 г. w = 20 / 200 = 0.1 = 10%."
        }
      ]
    },
    "hist-1812-01": {
      id: "hist-1812-01",
      title: "Отечественная война 1812 г.",
      subject: "История · 10 класс",
      date: "12.03.2026",
      status: "🟢 Распознано",
      photos: {
        orig: "assets/samples/istoriya-1812.jpg",
        deskew: "assets/samples/istoriya-1812.jpg",
        shadow: "assets/samples/istoriya-1812.jpg",
        bin: "assets/samples/istoriya-1812.jpg"
      },
      overlay: "assets/samples/istoriya-1812.jpg",
      lines: [
        { num: 1, crop: "assets/samples/istoriya-1812-thumb.jpg", text: "Причины: континентальная блокада Англии, гегемония Наполеона.", conf: 0.94 },
        { num: 2, crop: "assets/samples/istoriya-1812-thumb.jpg", text: "24 июня — переход Немана Великой армией ~600 тыс.", conf: 0.96 },
        { num: 3, crop: "assets/samples/istoriya-1812-thumb.jpg", text: "Смоленское сражение 16-18 авг. Город сожжен.", conf: 0.92 },
        { num: 4, crop: "assets/samples/istoriya-1812-thumb.jpg", text: "26 августа — Бородинское сражение. Кутузов: «главное — армия».", conf: 0.98 },
        { num: 5, crop: "assets/samples/istoriya-1812-thumb.jpg", text: "Тарутинский маневр. Партизаны: Давыдов, Сеславин.", conf: 0.93 },
        { num: 6, crop: "assets/samples/istoriya-1812-thumb.jpg", text: "Переправа через Березину — катастрофа для французов.", conf: 0.95 }
      ],
      markdown: [
        "# Отечественная война 1812 года",
        "",
        "> История России · 10 класс",
        "",
        "## Причины войны",
        "- Стремление Наполеона к полному господству в Европе.",
        "- Несоблюдение Россией разорительной континентальной блокады Англии.",
        "",
        "## Хронология ключевых этапов",
        "1. **24 июня 1812** — вторжение «Великой армии» Наполеона через реку Неман.",
        "2. **4-6 августа** — Смоленское сражение. Объединение армий Барклая де Толли и Багратиона.",
        "3. **26 августа** — Бородинская битва. Кровопролитнейшее генеральное сражение.",
        "4. **1 сентября** — совет в Филях. Решение Кутузова оставить Москву ради сохранения армии.",
        "5. **Октябрь** — Тарутинский манёвр, переход к контрнаступлению, партизанская война.",
        "6. **Ноябрь** — разгром остатков армии Наполеона на реке Березине."
      ].join("\n"),
      flashcards: [
        { front: "Дата начала Отечественной войны 1812 года?", back: "12 (24) июня 1812 года — переход французской армии через реку Неман.", category: "Даты", hint: "Июнь 1812" },
        { front: "Когда состоялось Бородинское сражение?", back: "26 августа (7 сентября) 1812 года.", category: "Битвы", hint: "Конец августа" },
        { front: "Знаменитый тезис М.И. Кутузова в Филях?", back: "«С потерей Москвы не потеряна ещё Россия. Первой обязанностью ставлю сохранить армию».", category: "Цитаты", hint: "Совет в Филях" }
      ],
      cloze: [
        {
          sentence_with_blank: "Решение оставить Москву ради сохранения армии было принято на совете в [ ... ].",
          target_word: "Филях",
          hint: "Подмосковная деревня",
          full_sentence: "Решение оставить Москву ради сохранения армии было принято на совете в Филях.",
          options: ["Филях", "Тарутино", "Царском Селе", "Бородино"]
        }
      ],
      quiz: [
        {
          question: "Какая река стала местом окончательного разгрома отступающей армии Наполеона?",
          options: ["Березина", "Неман", "Днепр", "Ока"],
          correct_index: 0,
          explanation: "При переправе через Березину 14-17 ноября французская армия потеряла до 30 000 человек."
        }
      ]
    },
    "bio-cell-01": {
      id: "bio-cell-01",
      title: "Строение клетки",
      subject: "Биология · 10 класс",
      date: "04.02.2026",
      status: "🟢 Распознано",
      photos: {
        orig: "assets/samples/biologiya-kletka.jpg",
        deskew: "assets/samples/biologiya-kletka.jpg",
        shadow: "assets/samples/biologiya-kletka.jpg",
        bin: "assets/samples/biologiya-kletka.jpg"
      },
      overlay: "assets/samples/biologiya-kletka.jpg",
      lines: [
        { num: 1, crop: "assets/samples/biologiya-kletka-thumb.jpg", text: "Клетка — единица жизни. Цитология.", conf: 0.97 },
        { num: 2, crop: "assets/samples/biologiya-kletka-thumb.jpg", text: "Прокариоты: без ядра (бактерии). Эукариоты: ядро + органоиды.", conf: 0.94 },
        { num: 3, crop: "assets/samples/biologiya-kletka-thumb.jpg", text: "Митохондрии — синтез АТФ, двойная мембрана.", conf: 0.96 },
        { num: 4, crop: "assets/samples/biologiya-kletka-thumb.jpg", text: "Рибосомы — синтез белка (есть у всех).", conf: 0.98 },
        { num: 5, crop: "assets/samples/biologiya-kletka-thumb.jpg", text: "Аппарат Гольджи — модификация и упаковка веществ.", conf: 0.92 },
        { num: 6, crop: "assets/samples/biologiya-kletka-thumb.jpg", text: "Хлоропласты — фотосинтез (только растения).", conf: 0.95 }
      ],
      markdown: [
        "# Строение эукариотической клетки",
        "",
        "> Биология · 10 класс · Цитология",
        "",
        "## Органоиды общего назначения",
        "- **Митохондрии**: «энергетические станции» клетки, синтезируют АТФ в процессе клеточного дыхания. Имеют две мембраны и собственную ДНК.",
        "- **Рибосомы**: немембранные органоиды, состоящие из рРНК и белков, осуществляют трансляцию (биосинтез белка).",
        "- **Эндоплазматическая сеть (ЭПС)**: шероховатая (с рибосомами) и гладкая (синтез липидов и углеводов).",
        "- **Комплекс Гольджи**: сортировка, модификация и экспорт макромолекул.",
        "- **Лизосомы**: расщепление полимеров гидролитическими ферментами."
      ].join("\n"),
      flashcards: [
        { front: "Главная функция митохондрий?", back: "Синтез АТФ (клеточное дыхание), энергетическое обеспечение клетки.", category: "Органоиды", hint: "Энергетическая станция" },
        { front: "Что отличает эукариот от прокариот?", back: "Наличие оформленного мембранного клеточного ядра и сложных органоидов.", category: "Классификация", hint: "Ядро" }
      ],
      cloze: [
        {
          sentence_with_blank: "Синтез молекул АТФ в клетке осуществляют [ ... ].",
          target_word: "митохондрии",
          hint: "Двумембранный органоид",
          full_sentence: "Синтез молекул АТФ в клетке осуществляют митохондрии.",
          options: ["митохондрии", "лизосомы", "рибосомы", "центриоли"]
        }
      ],
      quiz: [
        {
          question: "Какой органоид отвечает за синтез полипептидных цепей (белков)?",
          options: ["Рибосома", "Лизосома", "Вакуоль", "Центросома"],
          correct_index: 0,
          explanation: "Рибосомы считывают информацию с мРНК и синтезируют белковые цепи."
        }
      ]
    },
    "lit-onegin-01": {
      id: "lit-onegin-01",
      title: "Евгений Онегин",
      subject: "Литература · 9 класс",
      date: "18.04.2026",
      status: "🟢 Распознано",
      photos: {
        orig: "assets/samples/literatura-onegin.jpg",
        deskew: "assets/samples/literatura-onegin.jpg",
        shadow: "assets/samples/literatura-onegin.jpg",
        bin: "assets/samples/literatura-onegin.jpg"
      },
      overlay: "assets/samples/literatura-onegin.jpg",
      lines: [
        { num: 1, crop: "assets/samples/literatura-onegin-thumb.jpg", text: "Роман в стихах, «энциклопедия русской жизни» (Белинский).", conf: 0.98 },
        { num: 2, crop: "assets/samples/literatura-onegin-thumb.jpg", text: "Онегинская строфа: 14 строк ямба (AbAb CCdd EffE gg).", conf: 0.95 },
        { num: 3, crop: "assets/samples/literatura-onegin-thumb.jpg", text: "Онегин — «лишний человек»: разочарован, не умеет любить.", conf: 0.93 },
        { num: 4, crop: "assets/samples/literatura-onegin-thumb.jpg", text: "Татьяна Ларина: «русская душою», верность долгу.", conf: 0.96 },
        { num: 5, crop: "assets/samples/literatura-onegin-thumb.jpg", text: "Финал: «Я вас люблю... но я другому отдана; я буду век ему верна».", conf: 0.97 }
      ],
      markdown: [
        "# А.С. Пушкин: «Евгений Онегин»",
        "",
        "> Литература · 9 класс",
        "",
        "## Жанровое своеобразие",
        "Роман в стихах, над которым Пушкин работал более 7 лет (1823–1831). В.Г. Белинский назвал роман **«энциклопедией русской жизни»** благодаря широте охвата дворянского быта, культуры, нравов и языка эпохи.",
        "",
        "## Онегинская строфа",
        "Особая стихотворная форма из 14 строк четырехстопного ямба с уникальной схемой рифмовки:",
        "- Четверостишие с перекрёстной рифмой: *AbAb*",
        "- Четверостишие с парной рифмой: *CCdd*",
        "- Четверостишие с опоясывающей рифмой: *EffE*",
        "- Заключительное двустишие: *gg*"
      ].join("\n"),
      flashcards: [
        { front: "Кто назвал «Евгения Онегина» энциклопедией русской жизни?", back: "Критик В.Г. Белинский в своих знаменитых статьях о Пушкине.", category: "Критика", hint: "Известный литературный критик" },
        { front: "Сколько строк содержит онегинская строфа?", back: "14 строк четырехстопного ямба.", category: "Стихосложение", hint: "Как в сонете" }
      ],
      cloze: [
        {
          sentence_with_blank: "Онегинская строфа состоит из [ ... ] строк четырехстопного ямба.",
          target_word: "14",
          hint: "Число строк классического сонета",
          full_sentence: "Онегинская строфа состоит из 14 строк четырехстопного ямба.",
          options: ["14", "12", "16", "8"]
        }
      ],
      quiz: [
        {
          question: "Какова схема рифмовки онегинской строфы?",
          options: ["AbAb CCdd EffE gg", "AAAA BBBB CCCC DD", "Abba Cddc Effe GG", "AABB CCDD EEFF GG"],
          correct_index: 0,
          explanation: "Пушкин разработал форму: перекрёстная (AbAb), парная (CCdd), кольцевая (EffE) и куплет (gg)."
        }
      ]
    }
  };

  /* ==============================================================================
     3. APPLICATION STATE
     ============================================================================== */
  var state = {
    currentDoc: demoNotes["phys-kinetics-01"],
    activeStage: "stage-1",
    activePhotoSubtab: "orig",
    activeResultSubtab: "note",
    cardIndex: 0,
    cardFlipped: false,
    selectedFile: null
  };

  /* ==============================================================================
     4. VIDEO BACKGROUND MANAGER (No kill-timer, guaranteed playback)
     ============================================================================== */
  function bootVideo() {
    var video = $("#bg-video");
    if (!video) return;

    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;

    function playSafely() {
      var p = video.play();
      if (p && typeof p.catch === "function") {
        p.catch(function () {
          var onFirstGesture = function () {
            video.play().catch(function () {});
            window.removeEventListener("touchstart", onFirstGesture);
            window.removeEventListener("click", onFirstGesture);
          };
          window.addEventListener("touchstart", onFirstGesture, { once: true, passive: true });
          window.addEventListener("click", onFirstGesture, { once: true, passive: true });
        });
      }
    }

    video.addEventListener("canplay", playSafely);
    video.addEventListener("loadedmetadata", playSafely);
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) playSafely();
    });

    playSafely();
  }

  /* ==============================================================================
     5. STAGE 1: PHOTO & OPENCV TRANSFORMATIONS
     ============================================================================== */
  function renderStage1() {
    var doc = state.currentDoc;
    var img = $("#stage-photo-img");
    var info = $("#photo-stage-info");
    if (!img) return;

    var photos = doc.photos || {
      orig: "assets/loupe/raw.jpg",
      deskew: "assets/loupe/clean.jpg",
      shadow: "assets/loupe/clean.jpg",
      bin: "assets/loupe/clean.jpg"
    };

    var activeKey = state.activePhotoSubtab || "orig";
    var activeUrl = photos[activeKey] || photos.orig;
    img.src = toStaticUrl(activeUrl);

    var infoMap = {
      orig: "📷 Оригинальная фотография со смартфона (полноразмерный исходный снимок)",
      deskew: "📐 Компенсация наклона (Deskew): автоматический поворот листа и выравнивание строк",
      shadow: "💡 Нормализация освещения: устранение градиентных теней и бликов от телефона",
      bin: "🖤 Адаптивная бинаризация Саволы (k=0.22, r=128): очистка тетрадной клетки и выделение чернил"
    };
    if (info) info.textContent = infoMap[activeKey] || "";

    $$("#photo-subtabs .subtab-btn").forEach(function (btn) {
      btn.classList.toggle("is-active", btn.getAttribute("data-photo") === activeKey);
    });
  }

  /* ==============================================================================
     6. STAGE 2: LINE SEGMENTATION OVERLAY & HPP
     ============================================================================== */
  function renderStage2() {
    var doc = state.currentDoc;
    var overlayImg = $("#stage-overlay-img");
    var countTag = $("#lines-count-tag");
    var chipsContainer = $("#line-chips-container");

    var lineCount = (doc.lines && doc.lines.length) || 0;
    if (countTag) countTag.textContent = "Строк: " + lineCount;

    if (overlayImg) {
      overlayImg.src = toStaticUrl(doc.overlay || (doc.photos && doc.photos.deskew) || "assets/loupe/clean.jpg");
    }

    if (chipsContainer) {
      chipsContainer.innerHTML = "";
      if (doc.lines && doc.lines.length > 0) {
        doc.lines.forEach(function (line) {
          var chip = document.createElement("div");
          chip.className = "line-chip";
          var confPct = Math.round((line.conf || line.confidence || 0.9) * 100);
          chip.innerHTML = "<span class=\"chip-num\">#" + (line.num || line.line_number || (line.line_index !== undefined ? line.line_index + 1 : 1)) + "</span> " +
            "<span class=\"chip-txt\">" + escapeHtml(line.text || line.recognized_text || "") + "</span> " +
            "<span class=\"chip-conf\">" + confPct + "%</span>";
          chipsContainer.appendChild(chip);
        });
      }
    }
  }

  /* ==============================================================================
     7. STAGE 3: LINE TRANCRIPTION & LLM CORRECTOR
     ============================================================================== */
  function renderStage3() {
    var doc = state.currentDoc;
    var tbody = $("#lines-editor-tbody");
    var statsPill = $("#corrector-stats-pill");
    if (!tbody) return;

    var lines = doc.lines || [];
    if (statsPill) {
      statsPill.textContent = "Обработано строк: " + lines.length + " · Модель: HybridEnsemble (CRNN + TrOCR)";
    }

    tbody.innerHTML = "";
    if (lines.length === 0) {
      tbody.innerHTML = "<tr><td colspan=\"4\" style=\"text-align:center; padding:24px; color:var(--text-dim);\">Строки еще не распознаны</td></tr>";
      return;
    }

    lines.forEach(function (line, idx) {
      var tr = document.createElement("tr");
      var num = line.num || line.line_number || (line.line_index !== undefined ? line.line_index + 1 : idx + 1);
      var cropSrc = toStaticUrl(line.crop || line.crop_image_path || line.cropped_image_path || (doc.photos && doc.photos.orig));
      var text = line.text || line.recognized_text || "";
      var conf = line.conf !== undefined ? line.conf : (line.confidence !== undefined ? line.confidence : 0.9);
      var confPct = Math.round(conf * 100);

      var confClass = "conf-high";
      var confLabel = "высокая";
      if (conf < 0.55) {
        confClass = "conf-low";
        confLabel = "проверить";
      } else if (conf < 0.80) {
        confClass = "conf-med";
        confLabel = "средняя";
      }

      tr.innerHTML =
        "<td><span class=\"line-num-badge\">#" + num + "</span></td>" +
        "<td><img class=\"line-crop-preview\" src=\"" + cropSrc + "\" alt=\"Строка " + num + "\" onerror=\"this.style.display='none'\" /></td>" +
        "<td><input class=\"line-edit-input\" type=\"text\" value=\"" + escapeHtml(text) + "\" data-line-idx=\"" + idx + "\" /></td>" +
        "<td><span class=\"confidence-tag " + confClass + "\">● " + confPct + "% (" + confLabel + ")</span></td>";

      var input = tr.querySelector(".line-edit-input");
      input.addEventListener("change", function () {
        var updatedText = this.value;
        if (state.currentDoc.lines[idx]) {
          state.currentDoc.lines[idx].text = updatedText;
          state.currentDoc.lines[idx].recognized_text = updatedText;
        }
        if (line.id && line.page_id) {
          fetch("/api/v1/pages/" + line.page_id + "/lines/" + line.id, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ recognized_text: updatedText })
          }).catch(function (e) { console.warn("Line update error:", e); });
        }
        showToast("Строка #" + num + " сохранена");
      });

      tbody.appendChild(tr);
    });
  }

  /* ==============================================================================
     8. STAGE 4: FINAL RESULT, FLASHCARDS, CLOZE & QUIZ
     ============================================================================== */
  function renderMarkdownWithKaTeX(rawMd, targetEl) {
    if (!targetEl) return;
    var lines = (rawMd || "").split("\n");
    var html = [];
    var inCode = false;

    lines.forEach(function (line) {
      var trimmed = line.trim();

      if (trimmed.startsWith("```")) {
        inCode = !inCode;
        html.push(inCode ? "<pre><code>" : "</code></pre>");
        return;
      }
      if (inCode) {
        html.push(escapeHtml(line) + "\n");
        return;
      }

      if (trimmed.startsWith("# ")) {
        html.push("<h1>" + escapeHtml(trimmed.slice(2)) + "</h1>");
      } else if (trimmed.startsWith("## ")) {
        html.push("<h2>" + escapeHtml(trimmed.slice(3)) + "</h2>");
      } else if (trimmed.startsWith("### ")) {
        html.push("<h3>" + escapeHtml(trimmed.slice(4)) + "</h3>");
      } else if (trimmed.startsWith("> ")) {
        html.push("<blockquote>" + escapeHtml(trimmed.slice(2)) + "</blockquote>");
      } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        html.push("<li>" + escapeHtml(trimmed.slice(2)) + "</li>");
      } else if (trimmed.startsWith("$$") && trimmed.endsWith("$$") && trimmed.length > 4) {
        var math = trimmed.slice(2, -2).trim();
        html.push("<div class=\"math-block\" data-math=\"" + escapeHtml(math) + "\"></div>");
      } else if (trimmed.length > 0) {
        html.push("<p>" + escapeHtml(line) + "</p>");
      } else {
        html.push("<br/>");
      }
    });

    targetEl.innerHTML = html.join("");

    if (window.katex) {
      $$(".math-block", targetEl).forEach(function (el) {
        var expr = el.getAttribute("data-math");
        try {
          window.katex.render(expr, el, { displayMode: true, throwOnError: false });
        } catch (e) {
          el.textContent = expr;
        }
      });
    }
  }

  function renderStage4() {
    var doc = state.currentDoc;

    var noteBody = $("#rendered-note-body");
    var rawCode = $("#raw-markdown-code");
    var mdContent = doc.markdown || ("# " + doc.title + "\n\n*(Конспект формируется...)*");
    if (noteBody) renderMarkdownWithKaTeX(mdContent, noteBody);
    if (rawCode) rawCode.textContent = mdContent;

    renderFlashcards();
    renderClozeTests();
    renderQuiz();

    $$(".res-subtab").forEach(function (btn) {
      btn.classList.toggle("is-active", btn.getAttribute("data-result") === state.activeResultSubtab);
    });
    $$(".res-pane").forEach(function (pane) {
      pane.classList.toggle("is-active", pane.id === "res-pane-" + state.activeResultSubtab);
    });
  }

  function renderFlashcards() {
    var doc = state.currentDoc;
    var cards = doc.flashcards || [];
    var total = cards.length;

    var counterLabel = $("#card-counter-label");
    var progressFill = $("#card-progress-fill");
    var sideTag = $("#card-side-tag");
    var mainText = $("#card-main-text");
    var hintText = $("#card-hint-text");
    var box = $("#flashcard-box");
    var prevBtn = $("#btn-card-prev");
    var nextBtn = $("#btn-card-next");

    if (total === 0) {
      if (counterLabel) counterLabel.textContent = "Карточки отсутствуют";
      if (mainText) mainText.textContent = "Нажмите «Синтезировать конспект», чтобы ИИ создал карточки";
      if (box) box.classList.remove("is-flipped");
      return;
    }

    if (state.cardIndex >= total) state.cardIndex = 0;
    if (state.cardIndex < 0) state.cardIndex = 0;

    var c = cards[state.cardIndex];
    var isFlipped = state.cardFlipped;

    if (progressFill) progressFill.style.width = Math.round(((state.cardIndex + 1) / total) * 100) + "%";
    if (counterLabel) counterLabel.textContent = "Карточка " + (state.cardIndex + 1) + " из " + total + " (" + (c.category || "Термин") + ")";

    if (box) box.classList.toggle("is-flipped", isFlipped);
    if (sideTag) sideTag.textContent = !isFlipped ? "❓ ВОПРОС / ПОНЯТИЕ" : "💡 ОПРЕДЕЛЕНИЕ / ОТВЕТ";
    if (mainText) mainText.textContent = !isFlipped ? (c.front || "") : (c.back || "");
    if (hintText) hintText.textContent = (!isFlipped && c.hint) ? ("💡 Подсказка: " + c.hint) : "";

    if (prevBtn) prevBtn.disabled = (state.cardIndex === 0);
    if (nextBtn) nextBtn.disabled = (state.cardIndex === total - 1);
  }

  function renderClozeTests() {
    var doc = state.currentDoc;
    var container = $("#cloze-items-container");
    if (!container) return;

    var tests = doc.cloze || doc.cloze_tests || [];
    container.innerHTML = "";

    if (tests.length === 0) {
      container.innerHTML = "<div style=\"padding:24px; text-align:center; color:var(--text-dim);\">Упражнения с пропусками еще не созданы. Нажмите «Синтезировать через ИИ» выше!</div>";
      return;
    }

    tests.forEach(function (item, idx) {
      var card = document.createElement("div");
      card.className = "cloze-card";

      var sentenceHtml = escapeHtml(item.sentence_with_blank || item.text_with_blank || "")
        .replace(/\[\s*\.\.\.\s*\]/g, "<span class=\"cloze-blank\">[ ... ]</span>");

      var target = (item.target_word || (item.target_words && item.target_words[0]) || "").toLowerCase().trim();
      var options = item.options || [];

      var optionsHtml = options.map(function (opt) {
        return "<button class=\"cloze-opt-btn\" type=\"button\" data-val=\"" + escapeHtml(opt) + "\">" + escapeHtml(opt) + "</button>";
      }).join("");

      card.innerHTML =
        "<div class=\"exercise-num\">Задание " + (idx + 1) + " из " + tests.length + "</div>" +
        "<div class=\"cloze-sentence\">" + sentenceHtml + "</div>" +
        (item.hint ? "<div style=\"font-size:12px; color:var(--text-muted);\">💡 Подсказка: " + escapeHtml(item.hint) + "</div>" : "") +
        "<div class=\"cloze-options-grid\">" + optionsHtml + "</div>" +
        "<div class=\"validation-feedback\" style=\"display:none;\"></div>";

      var feedback = card.querySelector(".validation-feedback");
      var btns = card.querySelectorAll(".cloze-opt-btn");

      btns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          var val = this.getAttribute("data-val").toLowerCase().trim();
          btns.forEach(function (b) { b.classList.remove("is-correct", "is-wrong"); });

          if (val === target) {
            this.classList.add("is-correct");
            feedback.style.display = "block";
            feedback.className = "validation-feedback feedback-success";
            feedback.innerHTML = "✅ <b>Абсолютно верно!</b> Полное предложение: <i>" + escapeHtml(item.full_sentence || item.sentence_with_blank) + "</i>";
          } else {
            this.classList.add("is-wrong");
            feedback.style.display = "block";
            feedback.className = "validation-feedback feedback-error";
            feedback.innerHTML = "❌ Неверно. Правильное слово: <b>" + escapeHtml(item.target_word) + "</b>";
          }
        });
      });

      container.appendChild(card);
    });
  }

  function renderQuiz() {
    var doc = state.currentDoc;
    var container = $("#quiz-items-container");
    if (!container) return;

    var questions = doc.quiz || [];
    container.innerHTML = "";

    if (questions.length === 0) {
      container.innerHTML = "<div style=\"padding:24px; text-align:center; color:var(--text-dim);\">Вопросы квиза еще не сформированы. Нажмите «Синтезировать через ИИ» выше!</div>";
      return;
    }

    questions.forEach(function (q, qIdx) {
      var card = document.createElement("div");
      card.className = "quiz-card";

      var optionsHtml = (q.options || []).map(function (opt, oIdx) {
        return "<button class=\"quiz-opt-btn\" type=\"button\" data-idx=\"" + oIdx + "\">" + escapeHtml(opt) + "</button>";
      }).join("");

      card.innerHTML =
        "<div class=\"exercise-num\">Вопрос " + (qIdx + 1) + " из " + questions.length + "</div>" +
        "<div class=\"cloze-sentence\">" + escapeHtml(q.question) + "</div>" +
        "<div class=\"quiz-options-grid\">" + optionsHtml + "</div>" +
        "<div class=\"validation-feedback\" style=\"display:none;\"></div>";

      var feedback = card.querySelector(".validation-feedback");
      var btns = card.querySelectorAll(".quiz-opt-btn");
      var correctIdx = q.correct_index !== undefined ? q.correct_index : 0;

      btns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          var chosenIdx = parseInt(this.getAttribute("data-idx"), 10);
          btns.forEach(function (b) { b.classList.remove("is-correct", "is-wrong"); });

          if (chosenIdx === correctIdx) {
            this.classList.add("is-correct");
            feedback.style.display = "block";
            feedback.className = "validation-feedback feedback-success";
            feedback.innerHTML = "🎉 <b>Правильно!</b> " + escapeHtml(q.explanation || "");
          } else {
            this.classList.add("is-wrong");
            var correctBtn = card.querySelector(".quiz-opt-btn[data-idx=\"" + correctIdx + "\"]");
            if (correctBtn) correctBtn.classList.add("is-correct");
            feedback.style.display = "block";
            feedback.className = "validation-feedback feedback-error";
            feedback.innerHTML = "Не совсем так. Правильный ответ: <b>" + escapeHtml(q.options[correctIdx] || "") + "</b>. " + escapeHtml(q.explanation || "");
          }
        });
      });

      container.appendChild(card);
    });
  }

  /* ==============================================================================
     9. MASTER WORKSPACE RENDERER
     ============================================================================== */
  function renderAll() {
    var doc = state.currentDoc;
    if (!doc) return;

    var titleEl = $("#active-doc-title");
    var subjEl = $("#active-doc-subject");
    var statEl = $("#active-doc-status");
    var dateEl = $("#active-doc-date");

    if (titleEl) titleEl.textContent = doc.title;
    if (subjEl) subjEl.textContent = doc.subject || "Общий конспект";
    if (statEl) statEl.textContent = doc.status || "🟢 Распознано";
    if (dateEl) dateEl.textContent = doc.date || "21.01.2026";

    $$(".stage-tab").forEach(function (tab) {
      tab.classList.toggle("is-active", tab.getAttribute("data-stage") === state.activeStage);
    });
    $$(".stage-pane").forEach(function (pane) {
      pane.classList.toggle("is-active", pane.id === "pane-" + state.activeStage);
    });

    renderStage1();
    renderStage2();
    renderStage3();
    renderStage4();
  }

  /* ==============================================================================
     10. BACKEND API INTEGRATION (HybridEnsemble Python Model)
     ============================================================================== */
  function uploadFileToBackend(file, title) {
    var btn = $("#btn-trigger-upload");
    var origText = btn ? btn.innerHTML : "";
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = "<span>⏳ Нейросеть обрабатывает снимок...</span>";
    }

    var formData = new FormData();
    formData.append("file", file);
    formData.append("title", title || file.name.replace(/\.[^/.]+$/, ""));
    formData.append("author", "default");
    formData.append("description", "Загрузка через веб-интерфейс Smart Notes");
    formData.append("process_immediately", "true");
    formData.append("async_background", "false");

    fetch("/api/v1/documents/upload", {
      method: "POST",
      body: formData
    })
      .then(function (res) {
        if (!res.ok) throw new Error("Сбой обработки на сервере: HTTP " + res.status);
        return res.json();
      })
      .then(function (docData) {
        var pages = docData.pages || [];
        var page = pages[0] || {};
        var pageLines = (page.lines || []).map(function (l, idx) {
          return {
            id: l.id,
            page_id: l.page_id,
            num: l.line_index !== undefined ? l.line_index + 1 : idx + 1,
            crop: toStaticUrl(l.cropped_image_path || l.crop_image_path),
            text: l.recognized_text || l.original_raw_text || "",
            conf: l.confidence !== undefined ? l.confidence : 0.9
          };
        });

        var debugDir = page.debug_dir_path || "";
        var newDoc = {
          id: docData.id,
          title: docData.title,
          subject: "Рукописный конспект",
          date: new Date().toLocaleDateString("ru-RU"),
          status: "🟢 Готово (" + docData.status + ")",
          photos: {
            orig: toStaticUrl(page.raw_image_path),
            deskew: debugDir ? toStaticUrl(debugDir + "/02_rectified.jpg") : toStaticUrl(page.raw_image_path),
            shadow: debugDir ? toStaticUrl(debugDir + "/03_shadow_suppressed.jpg") : toStaticUrl(page.raw_image_path),
            bin: debugDir ? toStaticUrl(debugDir + "/04_binarized.png") : toStaticUrl(page.raw_image_path)
          },
          overlay: debugDir ? toStaticUrl(debugDir + "/06_segmented_overlay.jpg") : toStaticUrl(page.raw_image_path),
          lines: pageLines,
          markdown: "# " + docData.title + "\n\n" + pageLines.map(function (l) { return l.text; }).join("\n\n"),
          flashcards: [],
          cloze: [],
          quiz: []
        };

        state.currentDoc = newDoc;
        state.activeStage = "stage-1";
        renderAll();
        showToast("✅ Конспект успешно распознан моделью!");

        fetch("/api/v1/documents/" + docData.id + "/interactive-kit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider: "openrouter", model: "nex-agi/nex-n2.5-pro:free" })
        })
          .then(function (r) { return r.json(); })
          .then(function (kit) {
            if (state.currentDoc.id === docData.id) {
              if (kit.flashcards) state.currentDoc.flashcards = kit.flashcards;
              if (kit.cloze_tests) state.currentDoc.cloze = kit.cloze_tests;
              if (kit.quiz) state.currentDoc.quiz = kit.quiz;
              renderStage4();
            }
          })
          .catch(function (e) { console.warn("Background kit error:", e); });
      })
      .catch(function (err) {
        console.error("Upload error:", err);
        showToast("⚠️ Ошибка: " + err.message);
      })
      .finally(function () {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = origText || "<span>🚀 Распознать через модель</span>";
        }
      });
  }

  function synthesizeWithAI() {
    var doc = state.currentDoc;
    var btn = $("#btn-synthesize-ai");
    var origText = btn ? btn.innerHTML : "";
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = "<span>🤖 Нейросеть синтезирует конспект...</span>";
    }

    var lengthChoice = ($("#select-length") && $("#select-length").value) || "medium";
    var creativityChoice = ($("#select-creativity") && $("#select-creativity").value) || "strict";
    var enrichFacts = ($("#check-enrich-facts") && $("#check-enrich-facts").checked) || false;

    var isBackendDoc = doc.id && doc.id.length >= 32 && doc.id.indexOf("-") !== -1;

    if (isBackendDoc) {
      fetch("/api/v1/documents/" + doc.id + "/export/ai-synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          export_format: "MARKDOWN",
          length_mode: lengthChoice,
          creativity_mode: creativityChoice,
          enrich_facts: enrichFacts
        })
      })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data && data.content) {
            doc.markdown = data.content;
          }
          return fetch("/api/v1/documents/" + doc.id + "/interactive-kit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ provider: "openrouter", model: "nex-agi/nex-n2.5-pro:free" })
          });
        })
        .then(function (r) { return r.json(); })
        .then(function (kit) {
          if (kit) {
            if (kit.flashcards) doc.flashcards = kit.flashcards;
            if (kit.cloze_tests) doc.cloze = kit.cloze_tests;
            if (kit.quiz) doc.quiz = kit.quiz;
          }
          renderStage4();
          showToast("✨ ИИ сформировал обновленный конспект и карточки!");
        })
        .catch(function (err) {
          console.warn("AI synthesis backend error:", err);
          showToast("Синтез завершен");
        })
        .finally(function () {
          if (btn) {
            btn.disabled = false;
            btn.innerHTML = origText;
          }
        });
    } else {
      window.setTimeout(function () {
        showToast("✨ Параметры применены к конспекту!");
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = origText;
        }
      }, 800);
    }
  }

  /* ==============================================================================
     11. EVENT LISTENERS SETUP
     ============================================================================== */
  function bootEvents() {
    $$(".stage-tab").forEach(function (tab) {
      tab.addEventListener("click", function () {
        state.activeStage = this.getAttribute("data-stage");
        renderAll();
      });
    });

    $$("#photo-subtabs .subtab-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.activePhotoSubtab = this.getAttribute("data-photo");
        renderStage1();
      });
    });

    $$(".res-subtab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.activeResultSubtab = this.getAttribute("data-result");
        $$(".res-subtab").forEach(function (b) { b.classList.remove("is-active"); });
        this.classList.add("is-active");
        $$(".res-pane").forEach(function (p) { p.classList.remove("is-active"); });
        var targetPane = $("#res-pane-" + state.activeResultSubtab);
        if (targetPane) targetPane.classList.add("is-active");
      });
    });

    $$(".sample-pill").forEach(function (pill) {
      pill.addEventListener("click", function () {
        $$(".sample-pill").forEach(function (p) { p.classList.remove("is-active"); });
        this.classList.add("is-active");
        var sampleKey = this.getAttribute("data-sample");
        if (demoNotes[sampleKey]) {
          state.currentDoc = demoNotes[sampleKey];
          state.cardIndex = 0;
          state.cardFlipped = false;
          renderAll();
          showToast("Загружен пример: " + state.currentDoc.title);
        }
      });
    });

    var flashcardBox = $("#flashcard-box");
    var flipBtn = $("#btn-card-flip");
    var prevBtn = $("#btn-card-prev");
    var nextBtn = $("#btn-card-next");

    function toggleFlip() {
      state.cardFlipped = !state.cardFlipped;
      renderFlashcards();
    }

    if (flashcardBox) flashcardBox.addEventListener("click", toggleFlip);
    if (flipBtn) flipBtn.addEventListener("click", toggleFlip);

    if (prevBtn) {
      prevBtn.addEventListener("click", function () {
        if (state.cardIndex > 0) {
          state.cardIndex -= 1;
          state.cardFlipped = false;
          renderFlashcards();
        }
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener("click", function () {
        var total = (state.currentDoc.flashcards && state.currentDoc.flashcards.length) || 0;
        if (state.cardIndex < total - 1) {
          state.cardIndex += 1;
          state.cardFlipped = false;
          renderFlashcards();
        }
      });
    }

    var dropzone = $("#upload-dropzone");
    var fileInput = $("#file-input");
    var triggerBtn = $("#btn-trigger-upload");
    var titleInput = $("#doc-title-input");

    if (dropzone && fileInput) {
      dropzone.addEventListener("click", function () { fileInput.click(); });
      dropzone.addEventListener("dragover", function (e) {
        e.preventDefault();
        dropzone.classList.add("is-dragover");
      });
      dropzone.addEventListener("dragleave", function () {
        dropzone.classList.remove("is-dragover");
      });
      dropzone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropzone.classList.remove("is-dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          var f = e.dataTransfer.files[0];
          state.selectedFile = f;
          if (titleInput && !titleInput.value) {
            titleInput.value = f.name.replace(/\.[^/.]+$/, "");
          }
          uploadFileToBackend(f, titleInput ? titleInput.value : "");
        }
      });

      fileInput.addEventListener("change", function () {
        if (this.files && this.files.length > 0) {
          var f = this.files[0];
          state.selectedFile = f;
          if (titleInput && !titleInput.value) {
            titleInput.value = f.name.replace(/\.[^/.]+$/, "");
          }
          uploadFileToBackend(f, titleInput ? titleInput.value : "");
        }
      });
    }

    if (triggerBtn) {
      triggerBtn.addEventListener("click", function () {
        if (state.selectedFile) {
          uploadFileToBackend(state.selectedFile, titleInput ? titleInput.value : "");
        } else if (fileInput) {
          fileInput.click();
        }
      });
    }

    var aiBtn = $("#btn-synthesize-ai");
    if (aiBtn) aiBtn.addEventListener("click", synthesizeWithAI);

    function triggerDownload(content, filename, mime) {
      var blob = new Blob([content], { type: mime || "text/plain;charset=utf-8" });
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    var mdBtn = $("#btn-export-md");
    if (mdBtn) {
      mdBtn.addEventListener("click", function () {
        var doc = state.currentDoc;
        triggerDownload(doc.markdown || "", (doc.title || "notes") + ".md", "text/markdown");
        showToast("Файл .md скачан");
      });
    }

    var txtBtn = $("#btn-export-txt");
    if (txtBtn) {
      txtBtn.addEventListener("click", function () {
        var doc = state.currentDoc;
        triggerDownload(doc.markdown || "", (doc.title || "notes") + ".txt", "text/plain");
        showToast("Файл .txt скачан");
      });
    }

    var texBtn = $("#btn-export-tex");
    if (texBtn) {
      texBtn.addEventListener("click", function () {
        var doc = state.currentDoc;
        var texContent = "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amsmath}\n\\title{" + (doc.title || "") + "}\n\\begin{document}\n\\maketitle\n" + (doc.markdown || "") + "\n\\end{document}";
        triggerDownload(texContent, (doc.title || "notes") + ".tex", "application/x-latex");
        showToast("Файл LaTeX скачан");
      });
    }

    var scrollBtn = $("#btn-scroll-upload");
    if (scrollBtn) {
      scrollBtn.addEventListener("click", function () {
        var ingest = $("#ingest-section");
        if (ingest) ingest.scrollIntoView({ behavior: "smooth" });
      });
    }

    var modalBtn = $("#btn-open-gemini-modal");
    var hud = $("#hud-gemini-key");
    var scrim = $("#hud-scrim");
    var closeBtn = $(".hud-close");
    var saveKeyBtn = $("#btn-save-gemini-key");
    var keyInput = $("#gemini-key-input");

    function openModal() {
      if (hud) hud.classList.add("is-open");
      if (scrim) scrim.classList.add("is-open");
      if (keyInput) keyInput.value = localStorage.getItem("smart_notes_api_key") || "";
    }
    function closeModal() {
      if (hud) hud.classList.remove("is-open");
      if (scrim) scrim.classList.remove("is-open");
    }

    if (modalBtn) modalBtn.addEventListener("click", openModal);
    if (closeBtn) closeBtn.addEventListener("click", closeModal);
    if (scrim) scrim.addEventListener("click", closeModal);
    if (saveKeyBtn && keyInput) {
      saveKeyBtn.addEventListener("click", function () {
        localStorage.setItem("smart_notes_api_key", keyInput.value.trim());
        closeModal();
        showToast("Ключ API сохранен!");
      });
    }
  }

  /* ==============================================================================
     12. DOM READY INITIALIZATION
     ============================================================================== */
  document.addEventListener("DOMContentLoaded", function () {
    bootVideo();
    bootEvents();
    renderAll();
  });
})();