(function () {
  "use strict";

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatBytes(n) {
    if (!n || isNaN(n)) return "0 Б";
    if (n < 1024) return n + " Б";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " КБ";
    return (n / (1024 * 1024)).toFixed(2) + " МБ";
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

  var libraryItems = [
    {
      id: "physics",
      title: "Кинетическая энергия",
      topic: "Физика · 9 класс",
      date: "21.01.2026",
      bytes: 4800000,
      compressed: 86000,
      ratio: 98.2,
      status: "Вектор",
      kind: "demo",
      raw: "assets/loupe/raw.jpg",
      clean: "assets/loupe/clean.jpg",
      thumb: "assets/loupe/raw.jpg",
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
        "$$A = \\Delta E_k = \\frac{1200 \\times 20^2}{2} = 240\\,000\\,\\text{Дж} = 240\\,\\text{кДж}$$"
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
        }
      ]
    },
    {
      id: "istoriya-1812",
      title: "Отечественная война 1812 г.",
      topic: "История · 10 класс",
      date: "12.03.2026",
      bytes: 3400000,
      compressed: 110000,
      ratio: 96.7,
      status: "Вектор",
      kind: "demo",
      raw: "assets/samples/istoriya-1812.jpg",
      clean: null,
      thumb: "assets/samples/istoriya-1812-thumb.jpg",
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
        { front: "Когда состоялось Бородинское сражение?", back: "26 августа (7 сентября) 1812 года.", category: "Битвы", hint: "Конец августа" }
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
          explanation: "При переправе через Березину французская армия понесла катастрофические потери."
        }
      ]
    },
    {
      id: "literatura-onegin",
      title: "Евгений Онегин",
      topic: "Литература · 9 класс",
      date: "18.04.2026",
      bytes: 2900000,
      compressed: 94000,
      ratio: 96.7,
      status: "Вектор",
      kind: "demo",
      raw: "assets/samples/literatura-onegin.jpg",
      clean: null,
      thumb: "assets/samples/literatura-onegin-thumb.jpg",
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
        { num: 4, crop: "assets/samples/literatura-onegin-thumb.jpg", text: "Татьяна Ларина: «русская душою», верность долгу.", conf: 0.96 }
      ],
      markdown: [
        "# А.С. Пушкин: «Евгений Онегин»",
        "",
        "> Литература · 9 класс",
        "",
        "## Жанровое своеобразие",
        "Роман в стихах, который В.Г. Белинский назвал **«энциклопедией русской жизни»**.",
        "",
        "## Онегинская строфа",
        "14 строк четырехстопного ямба со схемой: AbAb CCdd EffE gg."
      ].join("\n"),
      flashcards: [
        { front: "Кто назвал роман «энциклопедией русской жизни»?", back: "В.Г. Белинский.", category: "Критика", hint: "Великий критик" }
      ],
      cloze: [
        {
          sentence_with_blank: "Онегинская строфа состоит из [ ... ] строк четырехстопного ямба.",
          target_word: "14",
          hint: "Число строк сонета",
          full_sentence: "Онегинская строфа состоит из 14 строк четырехстопного ямба.",
          options: ["14", "12", "16", "8"]
        }
      ],
      quiz: [
        {
          question: "Какова схема рифмовки онегинской строфы?",
          options: ["AbAb CCdd EffE gg", "AAAA BBBB CCCC DD", "Abba Cddc Effe GG", "AABB CCDD EEFF GG"],
          correct_index: 0,
          explanation: "Схема онегинской строфы: перекрестная, парная, кольцевая и куплет."
        }
      ]
    },
    {
      id: "biologiya-kletka",
      title: "Строение клетки",
      topic: "Биология · 9 класс",
      date: "04.02.2026",
      bytes: 3800000,
      compressed: 102000,
      ratio: 97.3,
      status: "Вектор",
      kind: "demo",
      raw: "assets/samples/biologiya-kletka.jpg",
      clean: null,
      thumb: "assets/samples/biologiya-kletka-thumb.jpg",
      photos: {
        orig: "assets/samples/biologiya-kletka.jpg",
        deskew: "assets/samples/biologiya-kletka.jpg",
        shadow: "assets/samples/biologiya-kletka.jpg",
        bin: "assets/samples/biologiya-kletka.jpg"
      },
      overlay: "assets/samples/biologiya-kletka.jpg",
      lines: [
        { num: 1, crop: "assets/samples/biologiya-kletka-thumb.jpg", text: "Клетка — единица жизни. Цитология.", conf: 0.97 },
        { num: 2, crop: "assets/samples/biologiya-kletka-thumb.jpg", text: "Прокариоты: без ядра. Эукариоты: ядро + органоиды.", conf: 0.94 },
        { num: 3, crop: "assets/samples/biologiya-kletka-thumb.jpg", text: "Митохондрии — синтез АТФ, двойная мембрана.", conf: 0.96 }
      ],
      markdown: [
        "# Строение эукариотической клетки",
        "",
        "> Биология · 9 класс · Цитология",
        "",
        "## Органоиды",
        "- **Митохондрии**: синтез АТФ.",
        "- **Рибосомы**: синтез белка."
      ].join("\n"),
      flashcards: [
        { front: "Функция митохондрий?", back: "Синтез АТФ, энергетическое обеспечение клетки.", category: "Органоиды", hint: "Энергетическая станция" }
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
          question: "Какой органоид отвечает за синтез белка?",
          options: ["Рибосома", "Лизосома", "Вакуоль", "Митохондрия"],
          correct_index: 0,
          explanation: "Рибосомы осуществляют трансляцию генетической информации в белок."
        }
      ]
    },
    {
      id: "himiya-rastvory",
      title: "Растворы и концентрации",
      topic: "Химия · 8 класс",
      date: "09.12.2025",
      bytes: 2600000,
      compressed: 88000,
      ratio: 96.6,
      status: "Вектор",
      kind: "demo",
      raw: "assets/samples/himiya-rastvory.jpg",
      clean: null,
      thumb: "assets/samples/himiya-rastvory-thumb.jpg",
      photos: {
        orig: "assets/samples/himiya-rastvory.jpg",
        deskew: "assets/samples/himiya-rastvory.jpg",
        shadow: "assets/samples/himiya-rastvory.jpg",
        bin: "assets/samples/himiya-rastvory.jpg"
      },
      overlay: "assets/samples/himiya-rastvory.jpg",
      lines: [
        { num: 1, crop: "assets/samples/himiya-rastvory-thumb.jpg", text: "Раствор = растворитель + растворенное вещество.", conf: 0.96 },
        { num: 2, crop: "assets/samples/himiya-rastvory-thumb.jpg", text: "Массовая доля: w = m(в-ва) / m(р-ра).", conf: 0.98 }
      ],
      markdown: [
        "# Растворы и концентрации",
        "",
        "> Химия · 8 класс",
        "",
        "## Массовая доля",
        "$$w = \\frac{m_{\\text{в-ва}}}{m_{\\text{р-ра}}}$$"
      ].join("\n"),
      flashcards: [
        { front: "Что такое массовая доля?", back: "Отношение массы вещества к общей массе раствора.", category: "Определение", hint: "w" }
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
          question: "В 180 г воды растворили 20 г соли. Чему равна массовая доля соли?",
          options: ["10%", "20%", "11.1%", "9%"],
          correct_index: 0,
          explanation: "Масса раствора 200 г, w = 20 / 200 = 10%."
        }
      ]
    }
  ];

  var currentDoc = libraryItems[0];
  var activePhotoSubtab = "orig";
  var cardIndex = 0;
  var cardFlipped = false;

  function setStatus(text) {
    var el = $("#loupe-optic-text");
    if (el) el.textContent = text;
  }

  function setSplit(pct) {
    pct = Math.max(6, Math.min(94, pct));
    var loupe = $("#loupe");
    if (loupe) loupe.style.setProperty("--split", pct + "%");
    return pct;
  }

  function setLayer(el, url) {
    if (!el || !url) return;
    el.style.backgroundImage = "url('" + url + "')";
  }

  function updateTelemetry(originalBytes, compressedBytes) {
    var cs = $("#chip-size");
    if (cs) cs.textContent = "[ СЖАТИЕ: " + formatBytes(originalBytes) + " → " + formatBytes(compressedBytes) + " ]";
  }

  function animateSplit(from, to, ms) {
    return new Promise(function (resolve) {
      var t0 = performance.now();
      var dur = ms || 1400;
      function frame(now) {
        var k = Math.min(1, (now - t0) / dur);
        var ease = 1 - Math.pow(1 - k, 3);
        setSplit(from + (to - from) * ease);
        if (k < 1) window.requestAnimationFrame(frame);
        else resolve();
      }
      window.requestAnimationFrame(frame);
    });
  }

  function pulseStatus(steps) {
    var i = 0;
    setStatus(steps[0]);
    var id = window.setInterval(function () {
      i += 1;
      if (i >= steps.length) { window.clearInterval(id); return; }
      setStatus(steps[i]);
    }, 520);
    return function stop() { window.clearInterval(id); };
  }

  function renderLibraryTable() {
    var tbody = $("#library-tbody");
    if (!tbody) return;
    tbody.innerHTML = libraryItems.map(function (item) {
      var badge = item.kind === "user" ? '<span class="doc-badge">Пользовательский</span>' : '<span class="doc-badge">Демо</span>';
      var pillClass = item.kind === "user" ? "status-pill is-user" : "status-pill";
      var thumb = item.thumb ? '<img class="doc-thumb" src="' + escapeHtml(item.thumb) + '" alt="">' : "";
      return (
        '<tr data-id="' + escapeHtml(item.id) + '">' +
          '<td><div class="doc-name">' + thumb + '<span>' + escapeHtml(item.title) + "</span>" + badge + "</div></td>" +
          "<td>" + escapeHtml(item.topic) + "</td>" +
          "<td>" + escapeHtml(item.date) + "</td>" +
          "<td>" + formatBytes(item.bytes) + "</td>" +
          "<td>" + (item.ratio ? item.ratio.toFixed(1) + "%" : "—") + "</td>" +
          '<td><span class="' + pillClass + '">' + escapeHtml(item.status) + "</span></td>" +
          '<td style="text-align:right;"><button class="row-action" type="button" data-open="' + escapeHtml(item.id) + '">Открыть</button></td>' +
        "</tr>"
      );
    }).join("");
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
    window.clearTimeout(showToast._t);
    showToast._t = window.setTimeout(function () { toast.classList.remove("is-open"); }, 2200);
  }

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

  function renderReaderPanes(item) {
    if (!item) return;

    // 1. Photo Stage
    var scanImg = $("#reader-scan-img");
    var photos = item.photos || { orig: item.raw, deskew: item.clean || item.raw, shadow: item.clean || item.raw, bin: item.clean || item.raw };
    if (scanImg) {
      scanImg.src = toStaticUrl(photos[activePhotoSubtab] || photos.orig || item.raw);
    }
    $$("#reader-photo-subtabs .subtab-btn").forEach(function (btn) {
      btn.classList.toggle("is-active", btn.getAttribute("data-photo") === activePhotoSubtab);
    });

    // 2. Line Segmentation Overlay
    var overlayImg = $("#reader-overlay-img");
    if (overlayImg) {
      overlayImg.src = toStaticUrl(item.overlay || (item.photos && item.photos.deskew) || item.clean || item.raw);
    }

    // 3. Lines Editor
    var linesTbody = $("#reader-lines-tbody");
    var linesCount = $("#reader-lines-count");
    var lines = item.lines || [];
    if (linesCount) linesCount.textContent = "Строк: " + lines.length;
    if (linesTbody) {
      linesTbody.innerHTML = "";
      if (lines.length === 0) {
        linesTbody.innerHTML = "<tr><td colspan=\"4\" style=\"text-align:center; padding:24px; color:#888;\">Строки модели еще не сформированы</td></tr>";
      } else {
        lines.forEach(function (line, idx) {
          var tr = document.createElement("tr");
          var num = line.num || (idx + 1);
          var cropSrc = toStaticUrl(line.crop || line.crop_image_path || line.cropped_image_path || item.raw);
          var text = line.text || line.recognized_text || "";
          var conf = line.conf !== undefined ? line.conf : (line.confidence !== undefined ? line.confidence : 0.9);
          var confPct = Math.round(conf * 100);
          var confClass = conf >= 0.8 ? "conf-high" : (conf >= 0.55 ? "conf-med" : "conf-low");

          tr.innerHTML =
            "<td><span class=\"line-num-badge\">#" + num + "</span></td>" +
            "<td><img class=\"line-crop-preview\" src=\"" + cropSrc + "\" alt=\"Строка " + num + "\" onerror=\"this.style.display='none'\" /></td>" +
            "<td><input class=\"line-edit-input\" type=\"text\" value=\"" + escapeHtml(text) + "\" data-line-idx=\"" + idx + "\" /></td>" +
            "<td><span class=\"confidence-tag " + confClass + "\">● " + confPct + "%</span></td>";

          var inp = tr.querySelector(".line-edit-input");
          inp.addEventListener("change", function () {
            var val = this.value;
            if (item.lines && item.lines[idx]) {
              item.lines[idx].text = val;
              item.lines[idx].recognized_text = val;
            }
            if (line.id && line.page_id) {
              fetch("/api/v1/pages/" + line.page_id + "/lines/" + line.id, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ recognized_text: val })
              }).catch(function (e) { console.warn("Line update failed:", e); });
            }
            showToast("Строка #" + num + " сохранена");
          });

          linesTbody.appendChild(tr);
        });
      }
    }

    // 4. Lecture Note Body
    var bodyEl = $("#reader-content-body");
    if (bodyEl) {
      var md = item.markdown || bodyToMarkdown(item);
      renderMarkdownWithKaTeX(md, bodyEl);
    }

    // 5. Flashcards
    renderReaderFlashcards(item);

    // 6. Cloze Tests
    renderReaderCloze(item);

    // 7. Quiz
    renderReaderQuiz(item);
  }

  function renderReaderFlashcards(item) {
    var cards = item.flashcards || [];
    var total = cards.length;
    var counter = $("#reader-cards-counter");
    var sideTag = $("#reader-card-side");
    var mainText = $("#reader-card-text");
    var hintText = $("#reader-card-hint");
    var box = $("#reader-flashcard-box");
    var prev = $("#btn-reader-card-prev");
    var next = $("#btn-reader-card-next");

    if (total === 0) {
      if (counter) counter.textContent = "Карточки формируются...";
      if (mainText) mainText.textContent = "Нажмите «Синтезировать через ИИ» для создания карточек";
      if (hintText) hintText.textContent = "";
      if (box) box.classList.remove("is-flipped");
      return;
    }

    if (cardIndex >= total) cardIndex = 0;
    if (cardIndex < 0) cardIndex = 0;

    var c = cards[cardIndex];
    if (counter) counter.textContent = "Карточка " + (cardIndex + 1) + " из " + total + " (" + (c.category || "Термин") + ")";
    if (box) box.classList.toggle("is-flipped", cardFlipped);
    if (sideTag) sideTag.textContent = !cardFlipped ? "❓ ВОПРОС / ПОНЯТИЕ" : "💡 ОПРЕДЕЛЕНИЕ / ОТВЕТ";
    if (mainText) mainText.textContent = !cardFlipped ? (c.front || "") : (c.back || "");
    if (hintText) hintText.textContent = (!cardFlipped && c.hint) ? ("💡 Подсказка: " + c.hint) : "";
    if (prev) prev.disabled = (cardIndex === 0);
    if (next) next.disabled = (cardIndex === total - 1);
  }

  function renderReaderCloze(item) {
    var container = $("#reader-cloze-container");
    if (!container) return;
    var tests = item.cloze || item.cloze_tests || [];
    container.innerHTML = "";

    if (tests.length === 0) {
      container.innerHTML = "<div style=\"padding:24px; text-align:center; color:#888;\">Упражнения с пропусками создаются через ИИ. Нажмите «Синтезировать через ИИ»!</div>";
      return;
    }

    tests.forEach(function (t, idx) {
      var card = document.createElement("div");
      card.className = "cloze-card";
      var sentenceHtml = escapeHtml(t.sentence_with_blank || t.text_with_blank || "")
        .replace(/\[\s*\.\.\.\s*\]/g, "<span class=\"cloze-blank\">[ ... ]</span>");
      var target = (t.target_word || (t.target_words && t.target_words[0]) || "").toLowerCase().trim();
      var options = t.options || [];

      var optionsHtml = options.map(function (opt) {
        return "<button class=\"cloze-opt-btn\" type=\"button\" data-val=\"" + escapeHtml(opt) + "\">" + escapeHtml(opt) + "</button>";
      }).join("");

      card.innerHTML =
        "<div class=\"exercise-num\">Задание " + (idx + 1) + " из " + tests.length + "</div>" +
        "<div class=\"cloze-sentence\">" + sentenceHtml + "</div>" +
        (t.hint ? "<div style=\"font-size:12px; color:#888; margin-bottom:8px;\">💡 Подсказка: " + escapeHtml(t.hint) + "</div>" : "") +
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
            feedback.innerHTML = "✅ <b>Верно!</b> " + escapeHtml(t.full_sentence || t.sentence_with_blank);
          } else {
            this.classList.add("is-wrong");
            feedback.style.display = "block";
            feedback.className = "validation-feedback feedback-error";
            feedback.innerHTML = "❌ Неверно. Правильное слово: <b>" + escapeHtml(t.target_word) + "</b>";
          }
        });
      });

      container.appendChild(card);
    });
  }

  function renderReaderQuiz(item) {
    var container = $("#reader-quiz-container");
    if (!container) return;
    var questions = item.quiz || [];
    container.innerHTML = "";

    if (questions.length === 0) {
      container.innerHTML = "<div style=\"padding:24px; text-align:center; color:#888;\">Вопросы теста генерируются через ИИ. Нажмите «Синтезировать через ИИ»!</div>";
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
            feedback.innerHTML = "Не совсем так. Ответ: <b>" + escapeHtml(q.options[correctIdx] || "") + "</b>. " + escapeHtml(q.explanation || "");
          }
        });
      });

      container.appendChild(card);
    });
  }

  function bodyToMarkdown(item) {
    if (item.markdown) return item.markdown;
    var nl = String.fromCharCode(10);
    var out = ["# " + item.title, "", "> " + item.topic + " · " + item.date, ""];
    (item.body || []).forEach(function (block) {
      if (block.t === "h2") out.push("## " + block.v, "");
      else if (block.t === "p") out.push(block.v, "");
      else if (block.t === "quote") out.push("> " + block.v, "");
      else if (block.t === "ul") {
        block.v.forEach(function (li) { out.push("- " + li); });
        out.push("");
      } else if (block.t === "formula") {
        out.push("**" + (block.label || "Формула") + "**", "", "$$ " + block.v + " $$", "");
        if (block.note) out.push(block.note, "");
      }
    });
    return out.join(nl);
  }

  function bodyToLatex(item) {
    var nl = String.fromCharCode(10);
    var out = [
      "\\documentclass[12pt]{article}",
      "\\usepackage[T2A]{fontenc}",
      "\\usepackage[utf8]{inputenc}",
      "\\usepackage[russian]{babel}",
      "\\usepackage{amsmath}",
      "\\title{" + item.title + "}",
      "\\date{" + item.date + "}",
      "\\begin{document}",
      "\\maketitle",
      "\\textit{" + item.topic + "}",
      ""
    ];
    if (item.markdown) {
      out.push(item.markdown);
    } else {
      (item.body || []).forEach(function (block) {
        if (block.t === "h2") out.push("\\section*{" + block.v + "}");
        else if (block.t === "p") out.push(block.v, "");
        else if (block.t === "quote") out.push("\\begin{quote}", block.v, "\\end{quote}", "");
        else if (block.t === "ul") {
          out.push("\\begin{itemize}");
          block.v.forEach(function (li) { out.push("  \\item " + li); });
          out.push("\\end{itemize}", "");
        } else if (block.t === "formula") {
          out.push("\\paragraph{" + (block.label || "Формула") + "}");
          out.push("\\begin{equation*}", block.v, "\\end{equation*}");
          if (block.note) out.push(block.note, "");
        }
      });
    }
    out.push("\\end{document}", "");
    return out.join(nl);
  }

  function openReader(item) {
    if (!item) return;
    currentDoc = item;
    closeHuds();
    var screen = $("#reader-screen");
    if (!screen) return;
    var ratio = item.ratio ? item.ratio.toFixed(1) + "%" : "—";
    var setText = function (sel, value) { var el = $(sel); if (el) el.textContent = value; };
    setText("#reader-doc-title", item.title);
    setText("#reader-doc-badge", item.topic + " · " + item.date);
    setText("#reader-stat-ratio", ratio);
    setText("#reader-stat-size", formatBytes(item.compressed || 0));
    setText("#reader-scan-meta", "RAW SCAN · " + formatBytes(item.bytes || 0));

    // Default pane
    var ws = $("#reader-workspace");
    if (ws) ws.setAttribute("data-pane", "scan");
    $$(".reader-tab").forEach(function (t) {
      var isScan = t.getAttribute("data-pane") === "scan";
      t.classList.toggle("is-active", isScan);
      t.setAttribute("aria-selected", isScan ? "true" : "false");
    });

    renderReaderPanes(item);

    screen.hidden = false;
    document.body.setAttribute("data-view", "reader");
    var back = $("#btn-reader-back");
    if (back) back.focus();
  }

  function closeReader(reopenLibrary) {
    var screen = $("#reader-screen");
    document.body.setAttribute("data-view", "home");
    if (screen) {
      window.setTimeout(function () {
        if (document.body.getAttribute("data-view") !== "reader") screen.hidden = true;
      }, 300);
    }
    if (reopenLibrary) openHud("hud-library", $('a[href="#library"]'));
  }

  function bootReader() {
    var back = $("#btn-reader-back");
    if (back) back.addEventListener("click", function () { closeReader(true); });

    var copy = $("#btn-copy-md");
    if (copy) copy.addEventListener("click", async function () {
      if (!currentDoc) return;
      var md = bodyToMarkdown(currentDoc);
      try {
        await navigator.clipboard.writeText(md);
        showToast("Скопировано!");
      } catch (e) {
        var ta = document.createElement("textarea");
        ta.value = md;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); showToast("Скопировано!"); }
        catch (err) { showToast("Не удалось скопировать"); }
        ta.remove();
      }
    });

    var tex = $("#btn-download-tex");
    if (tex) tex.addEventListener("click", function () {
      if (!currentDoc) return;
      var blob = new Blob([bodyToLatex(currentDoc)], { type: "application/x-tex" });
      var name = (currentDoc.id || "glyph-note") + ".tex";
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast("LaTeX сохранён");
    });

    // Zoomable image
    var wrap = $(".zoomable-wrap");
    if (wrap) wrap.addEventListener("click", function () { wrap.classList.toggle("is-zoomed"); });

    // Reader Tabs navigation
    var ws = $("#reader-workspace");
    $$(".reader-tab").forEach(function (tab) {
      tab.addEventListener("click", function () {
        var pane = this.getAttribute("data-pane");
        if (!pane || !ws) return;
        ws.setAttribute("data-pane", pane);
        $$(".reader-tab").forEach(function (t) {
          var active = t.getAttribute("data-pane") === pane;
          t.classList.toggle("is-active", active);
          t.setAttribute("aria-selected", active ? "true" : "false");
        });
      });
    });

    // Photo Subtabs (orig, deskew, shadow, bin)
    $$("#reader-photo-subtabs .subtab-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        activePhotoSubtab = this.getAttribute("data-photo");
        renderReaderPanes(currentDoc);
      });
    });

    // Flashcard Flip & Nav
    var box = $("#reader-flashcard-box");
    var flip = $("#btn-reader-card-flip");
    var prev = $("#btn-reader-card-prev");
    var next = $("#btn-reader-card-next");

    function toggleFlip() {
      cardFlipped = !cardFlipped;
      renderReaderFlashcards(currentDoc);
    }
    if (box) box.addEventListener("click", toggleFlip);
    if (flip) flip.addEventListener("click", toggleFlip);
    if (prev) prev.addEventListener("click", function () {
      if (cardIndex > 0) { cardIndex--; cardFlipped = false; renderReaderFlashcards(currentDoc); }
    });
    if (next) next.addEventListener("click", function () {
      var total = (currentDoc.flashcards && currentDoc.flashcards.length) || 0;
      if (cardIndex < total - 1) { cardIndex++; cardFlipped = false; renderReaderFlashcards(currentDoc); }
    });

    // AI Synthesize button in reader
    var synthBtn = $("#btn-reader-synthesize");
    if (synthBtn) {
      synthBtn.addEventListener("click", function () {
        if (!currentDoc) return;
        var btn = this;
        var origText = btn.textContent;
        btn.disabled = true;
        btn.textContent = "🤖 Синтез ИИ...";

        var len = ($("#reader-select-length") && $("#reader-select-length").value) || "medium";
        var cr = ($("#reader-select-creativity") && $("#reader-select-creativity").value) || "strict";
        var enrich = ($("#reader-check-enrich") && $("#reader-check-enrich").checked) || false;

        var isBackendDoc = currentDoc.id && currentDoc.id.length >= 32;

        if (isBackendDoc) {
          fetch("/api/v1/documents/" + currentDoc.id + "/export/ai-synthesize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ export_format: "MARKDOWN", length_mode: len, creativity_mode: cr, enrich_facts: enrich })
          })
            .then(function (r) { return r.json(); })
            .then(function (res) {
              if (res && res.content) currentDoc.markdown = res.content;
              return fetch("/api/v1/documents/" + currentDoc.id + "/interactive-kit", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ provider: "openrouter", model: "nex-agi/nex-n2.5-pro:free" })
              });
            })
            .then(function (r) { return r.json(); })
            .then(function (kit) {
              if (kit) {
                if (kit.flashcards) currentDoc.flashcards = kit.flashcards;
                if (kit.cloze_tests) currentDoc.cloze = kit.cloze_tests;
                if (kit.quiz) currentDoc.quiz = kit.quiz;
              }
              renderReaderPanes(currentDoc);
              showToast("✨ ИИ обновил конспект и материалы!");
            })
            .catch(function (e) {
              console.warn("AI synth error:", e);
              showToast("Синтез завершен");
            })
            .finally(function () {
              btn.disabled = false;
              btn.textContent = origText;
            });
        } else {
          window.setTimeout(function () {
            btn.disabled = false;
            btn.textContent = origText;
            showToast("✨ Параметры применены!");
          }, 800);
        }
      });
    }

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && document.body.getAttribute("data-view") === "reader") closeReader(false);
    });
  }

  // Real FastAPI Backend Document Upload
  async function ingestFile(file) {
    if (!file || !file.type || file.type.indexOf("image") !== 0) {
      setStatus("НУЖЕН СНИМОК");
      return;
    }

    var loupe = $("#loupe");
    if (loupe) loupe.classList.add("is-scanning");
    var stop = pulseStatus([
      "SCAN · NORMALIZE",
      "SCAN · CONTRAST",
      "SCAN · CRNN + TrOCR",
      "SCAN · SYNTHESIZING"
    ]);
    setSplit(8);

    var title = file.name.replace(/\.[^.]+$/, "") || "Новый конспект";
    var formData = new FormData();
    formData.append("file", file);
    formData.append("title", title);
    formData.append("author", "default");
    formData.append("process_immediately", "true");
    formData.append("async_background", "false");

    try {
      var resp = await fetch("/api/v1/documents/upload", {
        method: "POST",
        body: formData
      });

      if (!resp.ok) throw new Error("HTTP " + resp.status);
      var docData = await resp.json();

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
      var rawUrl = toStaticUrl(page.raw_image_path);
      var cleanUrl = debugDir ? toStaticUrl(debugDir + "/06_segmented_overlay.jpg") : rawUrl;

      setLayer($("#layer-raw"), rawUrl);
      setLayer($("#layer-clean"), cleanUrl);

      var compressedBytes = Math.round(file.size * 0.042);
      updateTelemetry(file.size, compressedBytes);

      var today = new Date();
      var dd = String(today.getDate()).padStart(2, "0");
      var mm = String(today.getMonth() + 1).padStart(2, "0");

      var newDoc = {
        id: docData.id,
        title: docData.title || title,
        topic: "Пользовательский скан",
        date: dd + "." + mm + "." + today.getFullYear(),
        bytes: file.size,
        compressed: compressedBytes,
        ratio: 95.8,
        status: "Вектор",
        kind: "user",
        raw: rawUrl,
        clean: cleanUrl,
        thumb: rawUrl,
        photos: {
          orig: rawUrl,
          deskew: debugDir ? toStaticUrl(debugDir + "/02_rectified.jpg") : rawUrl,
          shadow: debugDir ? toStaticUrl(debugDir + "/03_shadow_suppressed.jpg") : rawUrl,
          bin: debugDir ? toStaticUrl(debugDir + "/04_binarized.png") : rawUrl
        },
        overlay: cleanUrl,
        lines: pageLines,
        markdown: "# " + (docData.title || title) + "\n\n" + pageLines.map(function (l) { return l.text; }).join("\n\n"),
        flashcards: [],
        cloze: [],
        quiz: []
      };

      libraryItems.unshift(newDoc);
      renderLibraryTable();

      stop();
      await animateSplit(8, 88, 1400);
      if (loupe) loupe.classList.remove("is-scanning");
      setStatus("SPECTRAL 1200 DPI");

      // Open in Reader Studio
      openReader(newDoc);
      showToast("✅ Конспект успешно распознан моделью!");

      // Enrich kit in background
      fetch("/api/v1/documents/" + docData.id + "/interactive-kit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: "openrouter", model: "nex-agi/nex-n2.5-pro:free" })
      })
        .then(function (r) { return r.json(); })
        .then(function (kit) {
          if (newDoc.id === docData.id && kit) {
            if (kit.flashcards) newDoc.flashcards = kit.flashcards;
            if (kit.cloze_tests) newDoc.cloze = kit.cloze_tests;
            if (kit.quiz) newDoc.quiz = kit.quiz;
            if (currentDoc && currentDoc.id === newDoc.id) {
              renderReaderPanes(newDoc);
            }
          }
        })
        .catch(function (e) { console.warn("Background kit error:", e); });

    } catch (err) {
      console.error("Backend OCR error:", err);
      stop();
      if (loupe) loupe.classList.remove("is-scanning");
      setStatus("ОШИБКА РАСПОЗНАВАНИЯ");
      showToast("⚠️ Сбой распознавания: " + err.message);
    }
  }

  function openPicker() {
    var input = $("#file-input");
    if (input) input.click();
  }

  function bootDropzone() {
    var stage = $("#loupe-stage");
    var input = $("#file-input");
    var uploadBtn = $("#btn-upload");
    var libUploadBtn = $("#btn-lib-upload");

    if (uploadBtn) uploadBtn.addEventListener("click", openPicker);
    if (libUploadBtn) libUploadBtn.addEventListener("click", openPicker);

    if (stage) {
      stage.addEventListener("click", function (e) {
        if (e.target.closest("#splitter")) return;
        openPicker();
      });
      stage.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openPicker();
        }
      });
      stage.addEventListener("dragover", function (e) {
        e.preventDefault();
        stage.classList.add("is-dragover");
      });
      stage.addEventListener("dragleave", function () {
        stage.classList.remove("is-dragover");
      });
      stage.addEventListener("drop", function (e) {
        e.preventDefault();
        stage.classList.remove("is-dragover");
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
          ingestFile(e.dataTransfer.files[0]);
        }
      });
    }

    if (input) {
      input.addEventListener("change", function () {
        if (input.files && input.files.length) {
          ingestFile(input.files[0]);
          input.value = "";
        }
      });
    }
  }

  function bootSplitter() {
    var loupe = $("#loupe");
    var stage = $("#loupe-stage");
    var splitter = $("#splitter");
    if (!loupe || !stage || !splitter) return;
    var dragging = false;

    function moveAt(clientX) {
      var rect = stage.getBoundingClientRect();
      if (!rect.width) return;
      var rel = clientX - rect.left;
      var pct = (rel / rect.width) * 100;
      setSplit(pct);
    }

    splitter.addEventListener("pointerdown", function (e) {
      e.preventDefault();
      e.stopPropagation();
      dragging = true;
      splitter.setPointerCapture(e.pointerId);
    });

    window.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      moveAt(e.clientX);
    });

    window.addEventListener("pointerup", function () { dragging = false; });
    window.addEventListener("pointercancel", function () { dragging = false; });
  }

  function bootParallax() {
    var loupe = $("#loupe");
    var glare = $("#loupe-glare");
    if (!loupe || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    var ticking = false;
    var targetRx = 0, targetRy = 0, currentRx = 0, currentRy = 0;

    function update() {
      currentRx += (targetRx - currentRx) * 0.1;
      currentRy += (targetRy - currentRy) * 0.1;
      loupe.style.transform = "rotateX(" + currentRx.toFixed(2) + "deg) rotateY(" + currentRy.toFixed(2) + "deg)";
      if (Math.abs(targetRx - currentRx) > 0.05 || Math.abs(targetRy - currentRy) > 0.05) {
        window.requestAnimationFrame(update);
      } else {
        ticking = false;
      }
    }

    window.addEventListener("pointermove", function (e) {
      var w = window.innerWidth, h = window.innerHeight;
      var nx = (e.clientX / w) * 2 - 1;
      var ny = (e.clientY / h) * 2 - 1;
      targetRy = nx * 5;
      targetRx = -ny * 4;
      if (glare) {
        var gx = (e.clientX / w) * 100;
        var gy = (e.clientY / h) * 100;
        glare.style.transform = "translate(" + (gx - 50) * 0.25 + "%, " + (gy - 50) * 0.25 + "%)";
      }
      if (!ticking) {
        ticking = true;
        window.requestAnimationFrame(update);
      }
    }, { passive: true });
  }

  function openHud(id, triggerEl) {
    closeHuds();
    var hud = $("#" + id);
    var scrim = $("#hud-scrim");
    if (!hud || !scrim) return;
    hud.classList.add("is-open");
    scrim.classList.add("is-open");
    if (triggerEl) triggerEl.classList.add("is-active");
    var firstBtn = hud.querySelector("button, a");
    if (firstBtn) firstBtn.focus();
  }

  function closeHuds() {
    $$(".hud.is-open").forEach(function (h) { h.classList.remove("is-open"); });
    var scrim = $("#hud-scrim");
    if (scrim) scrim.classList.remove("is-open");
    $$(".nav-pill.is-active").forEach(function (p) { p.classList.remove("is-active"); });
  }

  function bootNav() {
    var burger = $("#burger-btn");
    var scrim = $("#menu-backdrop");
    function setMenu(open) {
      document.body.classList.toggle("menu-open", open);
      if (burger) {
        burger.setAttribute("aria-expanded", open ? "true" : "false");
        burger.classList.toggle("is-active", open);
      }
    }
    if (burger) {
      burger.addEventListener("click", function () {
        var next = !document.body.classList.contains("menu-open");
        setMenu(next);
      });
    }
    if (scrim) scrim.addEventListener("click", function () { setMenu(false); });

    $$(".site-nav a").forEach(function (link) {
      link.addEventListener("click", function (e) {
        var href = this.getAttribute("href");
        if (href && href.charAt(0) === "#") {
          e.preventDefault();
          var targetId = "hud-" + href.slice(1);
          openHud(targetId, this);
          setMenu(false);
        }
      });
    });

    $$(".hud-close").forEach(function (btn) {
      btn.addEventListener("click", closeHuds);
    });
    var hudScrim = $("#hud-scrim");
    if (hudScrim) hudScrim.addEventListener("click", closeHuds);

    ["#btn-open-library", "#btn-hero-library"].forEach(function (sel) {
      var el = $(sel);
      if (el) el.addEventListener("click", function () {
        var pill = $('a[href="#library"]');
        openHud("hud-library", pill);
        setMenu(false);
      });
    });

    var tbody = $("#library-tbody");
    if (tbody) {
      tbody.addEventListener("click", function (e) {
        var btn = e.target.closest("[data-open]");
        var row = e.target.closest("tr[data-id]");
        var id = (btn && btn.getAttribute("data-open")) || (row && row.getAttribute("data-id"));
        if (!id) return;
        var item = libraryItems.filter(function (it) { return it.id === id; })[0];
        openReader(item);
      });
    }
  }

  // Reliable Video Background Playback (No artificial timeouts)
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
      if (!document.hidden && document.body.getAttribute("data-view") !== "reader") {
        playSafely();
      }
    });

    playSafely();
  }

  function reveal() {
    var nodes = $$(".logo, .nav-pill, .header-cta, .badge, .line-inner, em, .hero-lede, .hero-actions .btn, .stat-item, .loupe-col");
    function markIn() {
      nodes.forEach(function (el) { el.classList.add("is-in"); });
    }
    nodes.forEach(function (el) {
      el.addEventListener("animationend", function () { el.classList.add("is-in"); });
    });
    var frames = 0;
    function fallback() {
      frames += 1;
      if (frames >= 2) { window.setTimeout(markIn, 2200); return; }
      window.requestAnimationFrame(fallback);
    }
    window.requestAnimationFrame(fallback);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) markIn();
  }

  document.addEventListener("DOMContentLoaded", function () {
    reveal();
    bootVideo();
    bootSplitter();
    bootParallax();
    bootNav();
    bootDropzone();
    bootReader();
    renderLibraryTable();
    setSplit(42);
  });
})();