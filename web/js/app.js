/**
 * Glyph.ai — Next-Gen Handwritten Document Neural Compiler
 * State Management, Reader Studio, KaTeX Rendering, Scan Zoom & Gemini Copilot
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

  function formatBytes(n) {
    if (window.GlyphEngine && GlyphEngine.formatBytes) return GlyphEngine.formatBytes(n);
    if (!n || isNaN(n)) return "0 Б";
    if (n < 1024) return n + " Б";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " КБ";
    return (n / (1024 * 1024)).toFixed(2) + " МБ";
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
    }, 2200);
  }

  /* ==============================================================================
     2. DEMO SMART NOTES DATASET (Strictly adheres to SmartNote schema)
     ============================================================================== */
  var demoNotes = [
    {
      id: "phys-kinetics-01",
      title: "Кинетическая энергия",
      subject: "Физика · 9 класс",
      date: "21.01.2026",
      rawUrl: "assets/loupe/raw.jpg",
      cleanUrl: "assets/loupe/clean.jpg",
      thumb: "assets/loupe/raw.jpg",
      rawBytes: 4800000,
      cleanBytes: 86000,
      compressionRatio: 98.2,
      rawOcrText: "Кинетическая энергия — энергия механического движения тела. Зависит от массы и квадрата скорости: E_k = m·v^2 / 2. Теорема об изменении: A = E_k2 - E_k1.",
      markdownBody: [
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
        "> Если работа сил положительна ($A > 0$), кинетическая энергия возрастает; если работа отрицательна ($A < 0$), кинетическая энергия убывает (например, при торможении под действием сил трения).",
        "",
        "## Важнейшие свойства",
        "- **Скалярность**: кинетическая энергия всегда строго неотрицательна ($E_k \\ge 0$), независимо от направления вектора скорости $\\vec{v}$.",
        "- **Относительность**: поскольку скорость $v$ зависит от выбора системы отсчёта, численное значение кинетической энергии также относительно.",
        "- **Квадратичная зависимость**: при увеличении скорости в $2$ раза кинетическая энергия возрастает в $4$ раза.",
        "",
        "## Решение типовой задачи с урока",
        "Автомобиль массой $m = 1200\\,\\text{кг}$ разгоняется из состояния покоя ($v_0 = 0$) до скорости $v = 20\\,\\text{м/с}$ ($72\\,\\text{км/ч}$). Найти работу двигателя по преодолению инерции:",
        "",
        "$$A = \\Delta E_k = \\frac{1200 \\times 20^2}{2} - 0 = \\frac{1200 \\times 400}{2} = 240\\,000\\,\\text{Дж} = 240\\,\\text{кДж}$$"
      ].join("\n"),
      isUserUploaded: false
    },
    {
      id: "phys-newton-01",
      title: "Законы Ньютона",
      subject: "Физика · 9 класс",
      date: "21.01.2026",
      rawUrl: "assets/samples/fizika-newton.jpg",
      cleanUrl: null,
      thumb: "assets/samples/fizika-newton-thumb.jpg",
      rawBytes: 4200000,
      cleanBytes: 95000,
      compressionRatio: 97.7,
      rawOcrText: "I закон: если F = 0, то v = const. II закон: F = ma. III закон: F1 = -F2. Вес P = mg, трение Fтр = mu*N. Задача: m=2кг, a=3м/с^2 -> F=6Н.",
      markdownBody: [
        "# Законы динамики Ньютона",
        "",
        "> Физика · 9 класс · Основы динамики",
        "",
        "## Первый закон Ньютона (Закон инерции)",
        "Существуют такие системы отсчёта, называемые инерциальными, в которых тело сохраняет состояние покоя или равномерного прямолинейного движения до тех пор, пока внешние воздействия не заставят его изменить это состояние.",
        "",
        "$$\\sum \\vec{F} = 0 \\implies \\vec{v} = \\text{const}$$",
        "",
        "## Второй закон Ньютона (Основной закон динамики)",
        "Ускорение тела прямо пропорционально равнодействующей всех приложенных к нему сил и обратно пропорционально массе этого тела:",
        "",
        "$$\\vec{a} = \\frac{\\sum \\vec{F}}{m} \\iff \\sum \\vec{F} = m \\vec{a}$$",
        "",
        "Единица силы в СИ — **Ньютон** ($\\text{Н}$):",
        "$$1\\,\\text{Н} = 1\\,\\text{кг} \\cdot \\text{м/с}^2$$",
        "",
        "## Третий закон Ньютона (Взаимодействие тел)",
        "Силы, с которыми два тела действуют друг на друга, равны по модулю, противоположны по направлению, направлены вдоль одной прямой и приложены к **разным** телам:",
        "",
        "$$\\vec{F}_{12} = -\\vec{F}_{21}$$",
        "",
        "## Ключевые силы механики",
        "- **Сила тяжести и вес**: $F_{\\text{тяж}} = mg$. Вес покоящегося на горизонтальной опоре тела $P = mg$. В свободном падении $N = 0$ (невесомость).",
        "- **Сила трения скольжения**: $F_{\\text{тр}} = \\mu N$, где $\\mu$ — безразмерный коэффициент трения, $N$ — сила нормальной реакции опоры.",
        "",
        "## Расчётная задача",
        "Тело массой $m = 2\\,\\text{кг}$ под действием постоянной горизонтальной силы приобретает ускорение $a = 3\\,\\text{м/с}^2$. Определить величину силы при отсутствии трения:",
        "",
        "$$F = m \\cdot a = 2\\,\\text{кг} \\times 3\\,\\text{м/с}^2 = 6\\,\\text{Н}$$"
      ].join("\n"),
      isUserUploaded: false
    },
    {
      id: "hist-1812-01",
      title: "Отечественная война 1812 г.",
      subject: "История · 10 класс",
      date: "12.03.2026",
      rawUrl: "assets/samples/istoriya-1812.jpg",
      cleanUrl: null,
      thumb: "assets/samples/istoriya-1812-thumb.jpg",
      rawBytes: 3400000,
      cleanBytes: 110000,
      compressionRatio: 96.7,
      rawOcrText: "Тема: Отечественная война 1812 года. Причины: континентальная блокада Англии, отказ России её соблюдать, гегемония Наполеона. 24 июня — Неман. Бородино 26 авг. Тарутинский манёвр. Партизаны.",
      markdownBody: [
        "# Отечественная война 1812 года",
        "",
        "> История России · 10 класс · Исторический очерк",
        "",
        "## Предпосылки и причины конфликта",
        "- **Континентальная блокада Англии**: условия Тильзитского мира 1807 года разрушали внешнюю торговлю России, вынуждая её де-факто допускать нейтральные суда.",
        "- **Стремление к гегемонии**: Наполеон стремился к единоличному господству в континентальной Европе и устранению последнего независимого центра силы.",
        "- **Польский вопрос**: создание Великого герцогства Варшавского воспринималось Александром I как прямая угроза границам империи.",
        "",
        "## Основные вехи кампании",
        "1. **24 июня 1812 г.** — Вторжение Великой армии (~$600\\,000$ солдат) через реку Неман без объявления войны.",
        "2. **План Барклая-де-Толли**: отступление 1-й и 2-й армий вглубь территории с целью соединения и сохранения боеспособности.",
        "3. **16–18 августа** — Смоленское сражение: соединение сил и оставление сожжённого города.",
        "4. **26 августа (7 сентября)** — Генеральное Бородинское сражение. Кутузов: *«Главное — сохранить армию»*. Потери французов превысили $50\\,000$ человек.",
        "5. **Военный совет в Филях**: историческое решение оставить Москву ради спасения России.",
        "6. **Тарутинский манёвр**: перекрытие Калужской дороги и отрезание врага от богатых южных губерний.",
        "7. **Партизанское движение**: армейские отряды Д. Давыдова, А. Сеславина, А. Фигнера и народные ополчения.",
        "8. **Ноябрь 1812 г.** — Катастрофа наполеоновской армии на реке Березине.",
        "",
        "## Исторические итоги",
        "- Полный разгром интервентов: из полумиллионной группировки спаслись лишь разрозненные остатки.",
        "- Начало Заграничных походов русской армии 1813–1814 гг. и освобождение Европы.",
        "- Мощный подъём национального самосознания, давший импульс культуре «Золотого века»."
      ].join("\n"),
      isUserUploaded: false
    },
    {
      id: "lit-onegin-01",
      title: "Евгений Онегин",
      subject: "Литература · 9 класс",
      date: "18.04.2026",
      rawUrl: "assets/samples/literatura-onegin.jpg",
      cleanUrl: null,
      thumb: "assets/samples/literatura-onegin-thumb.jpg",
      rawBytes: 2900000,
      cleanBytes: 94000,
      compressionRatio: 96.7,
      rawOcrText: "Роман в стихах, энциклопедия русской жизни. Онегинская строфа: 14 строк ямба, AbAb CCdd EffE gg. Онегин — лишний человек. Татьяна: русская душою. Я другому отдана.",
      markdownBody: [
        "# Роман в стихах «Евгений Онегин»",
        "",
        "> Литература · 9 класс · А. С. Пушкин",
        "",
        "## Жанровое новаторство",
        "А. С. Пушкин определил форму произведения как **роман в стихах** — сплав эпического сюжета с глубочайшим лирическим началом. В. Г. Белинский назвал роман «энциклопедией русской жизни».",
        "",
        "## Онегинская строфа",
        "Произведение написано особой $14$-строчной строфой четырёхстопного ямба со строгой структурой рифмовки:",
        "",
        "$$\\text{Схема строфы: } [AbAb] \\quad [CCdd] \\quad [EffE] \\quad [gg]$$",
        "",
        "- Первое четверостишие — **перекрёстная рифма** ($AbAb$);",
        "- Второе четверостишие — **парная рифма** ($CCdd$);",
        "- Третье четверостишие — **опоясывающая рифма** ($EffE$);",
        "- Заключительное двустишие — **афористическая кода** ($gg$).",
        "",
        "## Система образов",
        "- **Евгений Онегин**: архетип «лишнего человека» — образован, умён, но пресыщен и парализован хандрой. Страх перед светским мнением толкает его на дуэль с Ленским.",
        "- **Татьяна Ларина**: пушкинский нравственный идеал, цельная натура «с русскою душою». Знаменитое письмо к Онегину — образец искренней исповеди.",
        "- **Владимир Ленский**: поэт-романтик с идеалистическим взглядом на мир, чья гибель обнажает цинизм светских правил.",
        "",
        "## Кульминация и моральный выбор",
        "> «Я вас люблю (к чему лукавить?), / Но я другому отдана; / Я буду век ему верна.»",
        "",
        "В финале побеждает нравственный долг и верность собственному достоинству."
      ].join("\n"),
      isUserUploaded: false
    },
    {
      id: "bio-cell-01",
      title: "Строение клетки",
      subject: "Биология · 9 класс",
      date: "04.02.2026",
      rawUrl: "assets/samples/biologiya-kletka.jpg",
      cleanUrl: null,
      thumb: "assets/samples/biologiya-kletka-thumb.jpg",
      rawBytes: 3800000,
      cleanBytes: 102000,
      compressionRatio: 97.3,
      rawOcrText: "Клетка — единица жизни. Прокариоты и эукариоты. Органоиды: митохондрии (АТФ), рибосомы (белок), ЭПС, аппарат Гольджи, лизосомы, хлоропласты. Жидкостно-мозаичная модель 1972.",
      markdownBody: [
        "# Строение и физиология клетки",
        "",
        "> Биология · 9 класс · Цитология",
        "",
        "## Клетка как элементарная единица жизни",
        "**Клетка** — структурная, функциональная и генетическая единица всех живых организмов. Наука о клетке — **цитология**.",
        "",
        "## Две ветви клеточной организации",
        "- **Прокариоты** (бактерии, археи): лишены оформленного ядра, кольцевая молекула ДНК лежит в нуклеоиде, нет мембранных органоидов.",
        "- **Эукариоты** (растения, животные, грибы): имеют чётко оформленное ядро с оболочкой и хроматином, а также систему мембранных органелл.",
        "",
        "## Органоиды эукариотической клетки",
        "- **Плазматическая мембрана**: жидкостно-мозаичная модель (Сингер и Николсон, 1972). Липидный бислой со встроенными белками, обеспечивает избирательный транспорт.",
        "- **Митохондрии**: двумембранные «энергетические станции» с собственным геномом. Осуществляют синтез АТФ в процессе клеточного дыхания:",
        "$$\\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2 \\longrightarrow 6\\text{CO}_2 + 6\\text{H}_2\\text{O} + 36\\,\\text{АТФ}$$",
        "- **Рибосомы**: немембранные структуры, осуществляющие трансляцию (синтез белка из аминокислот).",
        "- **Эндоплазматическая сеть (ЭПС)**: шероховатая (синтез белков) и гладкая (синтез липидов и стероидов).",
        "- **Аппарат Гольджи**: сортировка, модификация и экспорт макромолекул в секреторных везикулах.",
        "- **Лизосомы**: гидролитические пузырьки для внутриклеточного расщепления биополимеров.",
        "- **Хлоропласты**: органоиды фотосинтеза в растительных клетках."
      ].join("\n"),
      isUserUploaded: false
    },
    {
      id: "chem-sol-01",
      title: "Растворы и концентрации",
      subject: "Химия · 8 класс",
      date: "09.12.2025",
      rawUrl: "assets/samples/himiya-rastvory.jpg",
      cleanUrl: null,
      thumb: "assets/samples/himiya-rastvory-thumb.jpg",
      rawBytes: 2600000,
      cleanBytes: 88000,
      compressionRatio: 96.6,
      rawOcrText: "Раствор = растворитель + в-во. Массовая доля w = m(в-ва)/m(р-ра)*100%. Молярность C = v/V. Кристаллогидраты: CuSO4*5H2O. Задача: 20г соли в 180г воды -> w=10%.",
      markdownBody: [
        "# Растворы и способы выражения концентрации",
        "",
        "> Химия · 8 класс · Физическая химия растворов",
        "",
        "## Определение раствора",
        "**Раствор** — термодинамически устойчивая гомогенная система переменного состава, состоящая из растворителя, растворённых веществ и продуктов их взаимодействия.",
        "",
        "$$\\text{Масса раствора: } m_{\\text{р-ра}} = m(\\text{в-ва}) + m(\\text{растворителя})$$",
        "",
        "## Массовая доля растворённого вещества (процентная концентрация)",
        "Отношение массы растворённого вещества к общей массе раствора:",
        "",
        "$$\\omega = \\frac{m_{\\text{в-ва}}}{m_{\\text{р-ра}}} \\times 100\\%$$",
        "",
        "## Молярная концентрация (молярность)",
        "Количество растворённого вещества ($\\nu$) в единице объёма раствора ($V$):",
        "",
        "$$C_M = \\frac{\\nu}{V} = \\frac{m}{M \\cdot V} \\quad \\left[\\frac{\\text{моль}}{\\text{л}}\\right]$$",
        "",
        "## Кристаллогидраты",
        "Вещества, содержащие в составе кристаллической решётки молекулы воды:",
        "$$\\text{CuSO}_4 \\cdot 5\\text{H}_2\\text{O} \\quad (\\text{пентагидрат сульфата меди II, медный купорос})$$",
        "",
        "## Расчётная задача с урока",
        "В $180\\,\\text{г}$ воды растворили $20\\,\\text{г}$ хлорида натрия $\\text{NaCl}$. Найти массовую долю соли:",
        "",
        "$$m_{\\text{р-ра}} = 20\\,\\text{г} + 180\\,\\text{г} = 200\\,\\text{г}$$",
        "$$\\omega(\\text{NaCl}) = \\frac{20\\,\\text{г}}{200\\,\\text{г}} = 0{,}10 = 10\\%$$"
      ].join("\n"),
      isUserUploaded: false
    }
  ];

  /* ==============================================================================
     3. USER NOTES LOCALSTORAGE REPOSITORY
     ============================================================================== */
  var STORAGE_USER_NOTES_KEY = "glyph_user_notes";
  var STORAGE_GEMINI_KEY = "glyph_gemini_api_key";

  function loadUserNotes() {
    try {
      var raw = localStorage.getItem(STORAGE_USER_NOTES_KEY);
      if (!raw) return [];
      var parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      console.warn("Не удалось прочитать пользовательские конспекты из localStorage:", e);
      return [];
    }
  }

  function saveUserNote(note) {
    try {
      var existing = loadUserNotes();
      existing.unshift(note);
      if (existing.length > 30) existing = existing.slice(0, 30);
      localStorage.setItem(STORAGE_USER_NOTES_KEY, JSON.stringify(existing));
    } catch (e) {
      console.warn("Не удалось сохранить конспект в localStorage:", e);
    }
  }

  function getAllNotes() {
    var userNotes = loadUserNotes();
    return userNotes.concat(demoNotes);
  }

  /* ==============================================================================
     4. MARKDOWN + KATEX PARSER & RENDERER
     ============================================================================== */
  function renderMarkdownToHtml(md) {
    if (!md) return "";
    var mathBlocks = [];
    var mathInlines = [];

    // 1. Extract block math $$...$$
    var text = String(md).replace(/\$\$([\s\S]*?)\$\$/g, function (_, tex) {
      var idx = mathBlocks.length;
      mathBlocks.push(tex.trim());
      return "@@@MATH_BLOCK_" + idx + "@@@";
    });

    // 2. Extract inline math $...$
    text = text.replace(/\$([^\$\n\r]+?)\$/g, function (_, tex) {
      var idx = mathInlines.length;
      mathInlines.push(tex.trim());
      return "@@@MATH_INLINE_" + idx + "@@@";
    });

    // 3. Process lines
    var lines = text.split("\n");
    var out = [];
    var inUl = false;
    var inOl = false;
    var inTable = false;
    var tableRows = [];

    function closeLists() {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (inOl) { out.push("</ol>"); inOl = false; }
    }

    function flushTable() {
      if (!inTable) return;
      inTable = false;
      if (!tableRows.length) return;
      var html = ["<table>"];
      tableRows.forEach(function (row, idx) {
        if (idx === 1 && /^[\s|:-]+$/.test(row.raw)) return;
        var cells = row.cells;
        var tag = idx === 0 ? "th" : "td";
        html.push("<tr>" + cells.map(function (c) {
          return "<" + tag + ">" + renderInline(c) + "</" + tag + ">";
        }).join("") + "</tr>");
      });
      html.push("</table>");
      out.push(html.join(""));
      tableRows = [];
    }

    function renderInline(str) {
      var s = escapeHtml(str);
      s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
      s = s.replace(/`([^`]+)`/g, "<code>$1</code>");

      // Replace inline math placeholders
      s = s.replace(/@@@MATH_INLINE_(\d+)@@@/g, function (_, idx) {
        var tex = mathInlines[parseInt(idx, 10)] || "";
        if (window.katex && typeof window.katex.renderToString === "function") {
          try {
            return window.katex.renderToString(tex, { displayMode: false, throwOnError: false });
          } catch (e) {
            return '<span class="katex-fallback">$' + escapeHtml(tex) + '$</span>';
          }
        }
        return '<span class="katex-fallback" data-tex="' + escapeHtml(tex) + '">$' + escapeHtml(tex) + '$</span>';
      });

      return s;
    }

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      var trimmed = line.trim();

      // Table row
      if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
        closeLists();
        inTable = true;
        var cells = trimmed.slice(1, -1).split("|").map(function (c) { return c.trim(); });
        tableRows.push({ raw: trimmed, cells: cells });
        continue;
      } else if (inTable) {
        flushTable();
      }

      // Empty line
      if (!trimmed) {
        closeLists();
        continue;
      }

      // Math block placeholder
      var mbMatch = trimmed.match(/^@@@MATH_BLOCK_(\d+)@@@$/);
      if (mbMatch) {
        closeLists();
        var mbIdx = parseInt(mbMatch[1], 10);
        var mbTex = mathBlocks[mbIdx] || "";
        var renderedBlock = "";
        if (window.katex && typeof window.katex.renderToString === "function") {
          try {
            renderedBlock = '<div class="katex-block">' + window.katex.renderToString(mbTex, { displayMode: true, throwOnError: false }) + '</div>';
          } catch (e) {
            renderedBlock = '<div class="katex-block"><code>' + escapeHtml(mbTex) + '</code></div>';
          }
        } else {
          renderedBlock = '<div class="katex-block" data-tex="' + escapeHtml(mbTex) + '"><code>' + escapeHtml(mbTex) + '</code></div>';
        }
        out.push(renderedBlock);
        continue;
      }

      // Headers
      if (trimmed.startsWith("### ")) {
        closeLists();
        out.push("<h3>" + renderInline(trimmed.slice(4)) + "</h3>");
        continue;
      }
      if (trimmed.startsWith("## ")) {
        closeLists();
        out.push("<h2>" + renderInline(trimmed.slice(3)) + "</h2>");
        continue;
      }
      if (trimmed.startsWith("# ")) {
        closeLists();
        out.push("<h1>" + renderInline(trimmed.slice(2)) + "</h1>");
        continue;
      }

      // Blockquotes
      if (trimmed.startsWith("> ")) {
        closeLists();
        out.push("<blockquote>" + renderInline(trimmed.slice(2)) + "</blockquote>");
        continue;
      }

      // Lists
      var ulMatch = trimmed.match(/^[-*•]\s+(.*)$/);
      if (ulMatch) {
        if (inOl) { out.push("</ol>"); inOl = false; }
        if (!inUl) { out.push("<ul>"); inUl = true; }
        out.push("<li>" + renderInline(ulMatch[1]) + "</li>");
        continue;
      }

      var olMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (olMatch) {
        if (inUl) { out.push("</ul>"); inUl = false; }
        if (!inOl) { out.push("<ol>"); inOl = true; }
        out.push("<li>" + renderInline(olMatch[1]) + "</li>");
        continue;
      }

      closeLists();

      // Regular paragraph
      var pContent = trimmed.replace(/@@@MATH_BLOCK_(\d+)@@@/g, function (_, idx) {
        var tex = mathBlocks[parseInt(idx, 10)] || "";
        if (window.katex && typeof window.katex.renderToString === "function") {
          try {
            return '<div class="katex-block">' + window.katex.renderToString(tex, { displayMode: true, throwOnError: false }) + '</div>';
          } catch (e) {
            return '<div class="katex-block"><code>' + escapeHtml(tex) + '</code></div>';
          }
        }
        return '<div class="katex-block"><code>' + escapeHtml(tex) + '</code></div>';
      });

      out.push("<p>" + renderInline(pContent) + "</p>");
    }

    closeLists();
    flushTable();

    return out.join("\n");
  }

  function ensureKatexHydration(container) {
    if (!container) return;
    if (window.katex && typeof window.katex.renderToString === "function") {
      var blocks = container.querySelectorAll(".katex-block[data-tex]");
      blocks.forEach(function (el) {
        var tex = el.getAttribute("data-tex");
        if (tex) {
          try {
            el.innerHTML = window.katex.renderToString(tex, { displayMode: true, throwOnError: false });
            el.removeAttribute("data-tex");
          } catch (e) {}
        }
      });
      var inlines = container.querySelectorAll(".katex-fallback[data-tex]");
      inlines.forEach(function (el) {
        var tex = el.getAttribute("data-tex");
        if (tex) {
          try {
            var span = document.createElement("span");
            span.innerHTML = window.katex.renderToString(tex, { displayMode: false, throwOnError: false });
            el.replaceWith(span);
          } catch (e) {}
        }
      });
    } else {
      window.setTimeout(function () { ensureKatexHydration(container); }, 250);
    }
  }

  /* ==============================================================================
     5. LATEX & MARKDOWN EXPORT HELPERS
     ============================================================================== */
  function noteToLatex(note) {
    var nl = String.fromCharCode(10);
    var cleanBody = (note.markdownBody || "")
      .replace(/#+\s+(.*)/g, "\\section*{$1}")
      .replace(/\*\*([^*]+)\*\*/g, "\\textbf{$1}")
      .replace(/\*([^*]+)\*/g, "\\textit{$1}")
      .replace(/`([^`]+)`/g, "\\texttt{$1}")
      .replace(/>\s+(.*)/g, "\\begin{quote}$1\\end{quote}");

    return [
      "\\documentclass[12pt,a4paper]{article}",
      "\\usepackage[T2A]{fontenc}",
      "\\usepackage[utf8]{inputenc}",
      "\\usepackage[russian]{babel}",
      "\\usepackage{amsmath,amssymb,amsfonts}",
      "\\usepackage{geometry}",
      "\\geometry{margin=2cm}",
      "",
      "\\title{\\textbf{" + (note.title || "Конспект") + "}}",
      "\\author{Glyph.ai Neural OCR}",
      "\\date{" + (note.date || "2026") + "}",
      "",
      "\\begin{document}",
      "\\maketitle",
      "\\textit{" + (note.subject || "Предмет") + "}",
      "\\hrulefill",
      "\\vspace{1em}",
      "",
      cleanBody,
      "",
      "\\end{document}",
      ""
    ].join(nl);
  }

  /* ==============================================================================
     6. GEMINI NEURAL NOTE PROCESSOR (Online API + Offline Realistic Mocks)
     ============================================================================== */
  class GeminiNoteProcessor {
    constructor() {
      this.apiKey = localStorage.getItem(STORAGE_GEMINI_KEY) || "";
      this.model = "gemini-1.5-flash";
      this.apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent";
    }

    setApiKey(key) {
      this.apiKey = (key || "").trim();
      if (this.apiKey) {
        localStorage.setItem(STORAGE_GEMINI_KEY, this.apiKey);
      } else {
        localStorage.removeItem(STORAGE_GEMINI_KEY);
      }
    }

    getApiKey() {
      return this.apiKey;
    }

    hasApiKey() {
      return Boolean(this.apiKey && this.apiKey.length > 10);
    }

    async processNote(note, promptType, customPrompt) {
      if (this.hasApiKey()) {
        try {
          return await this.callGeminiApi(note, promptType, customPrompt);
        } catch (err) {
          console.warn("Ошибка Gemini API, переключение на локальный методический движок:", err);
          return {
            text: this.getOfflineResponse(note, promptType, customPrompt),
            isOffline: true,
            networkError: true
          };
        }
      } else {
        return {
          text: this.getOfflineResponse(note, promptType, customPrompt),
          isOffline: true
        };
      }
    }

    async callGeminiApi(note, promptType, customPrompt) {
      var systemPrompt = "";
      if (promptType === "summary") {
        systemPrompt = "Ты — академический тьютор. Составь ёмкий, структурированный пересказ конспекта «" + note.title + "» (" + note.subject + "). Выдели главную мысль, 3-4 ключевых тезиса и фундаментальные формулы в формате KaTeX ($...$ или $$...$$). Отвечай на русском языке в чистом Markdown.";
      } else if (promptType === "exam_questions") {
        systemPrompt = "Ты — строгий экзаменатор. Составь 4 проверочных вопроса к экзамену по теме «" + note.title + "» (" + note.subject + ") с подробными эталонными ответами и формулами в KaTeX ($...$ / $$...$$). Форматируй ответ в Markdown.";
      } else if (promptType === "glossary") {
        systemPrompt = "Ты — научный редактор. Составь исчерпывающий глоссарий терминов, величин, формул и законов из конспекта «" + note.title + "» (" + note.subject + "). Дай точные академические определения с формулами в KaTeX. Форматируй ответ в Markdown.";
      } else {
        systemPrompt = "Ты — персональный AI-сопроцессор конспектов Glyph.ai. Ответь на вопрос учащегося по конспекту «" + note.title + "» (" + note.subject + "): «" + customPrompt + "». Опирайся на материал конспекта, используй научную точность и KaTeX формулы.";
      }

      var userText = [
        systemPrompt,
        "",
        "--- ТЕКСТ КОНСПЕКТА ---",
        note.markdownBody || note.rawOcrText
      ].join("\n");

      var endpoint = this.apiUrl + "?key=" + encodeURIComponent(this.apiKey);
      var response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: userText }] }],
          generationConfig: {
            temperature: 0.2,
            maxOutputTokens: 1024
          }
        })
      });

      if (!response.ok) {
        throw new Error("HTTP Status " + response.status);
      }

      var json = await response.json();
      var candidate = json.candidates && json.candidates[0];
      var text = candidate && candidate.content && candidate.content.parts && candidate.content.parts[0] && candidate.content.parts[0].text;
      if (!text) throw new Error("Пустой ответ от Gemini");

      return { text: text, isOffline: false };
    }

    getOfflineResponse(note, promptType, customPrompt) {
      // High-precision curated academic mocks for demo notes
      var id = note.id || "";
      if (id === "phys-kinetics-01") {
        if (promptType === "summary") {
          return [
            "### Краткий пересказ: Кинетическая энергия",
            "",
            "- **Сущность**: Кинетическая энергия $E_k$ — скалярная характеристика движения тела, пропорциональная его массе и квадрату скорости: $$E_k = \\frac{m v^2}{2}$$",
            "- **Теорема о работе**: Изменение кинетической энергии в точности равно совершённой механической работе всех сил: $A = \\Delta E_k = E_{k2} - E_{k1}$.",
            "- **Ключевой вывод**: При удвоении скорости энергия возрастает в $4$ раза, что определяет резкое увеличение тормозного пути транспорта."
          ].join("\n");
        }
        if (promptType === "exam_questions") {
          return [
            "### Экзаменационные вопросы и ответы",
            "",
            "1. **Может ли кинетическая энергия тела быть отрицательной?**",
            "   *Ответ*: Нет, поскольку масса $m > 0$, а скорость входит в формулу во второй степени ($v^2 \\ge 0$).",
            "",
            "2. **Как изменится $E_k$, если скорость тела уменьшится в 3 раза?**",
            "   *Ответ*: Уменьшится в $3^2 = 9$ раз.",
            "",
            "3. **В чём заключается теорема о кинетической энергии?**",
            "   *Ответ*: Работа всех приложенных сил равна изменению кинетической энергии: $A = \\Delta E_k$."
          ].join("\n");
        }
        if (promptType === "glossary") {
          return [
            "### Глоссарий определений",
            "",
            "- **Кинетическая энергия ($E_k$)** — скалярная величина, мера механического движения тела ($[\\text{Дж}]$).",
            "- **Механическая работа ($A$)** — процесс передачи энергии при перемещении тела под действием силы ($A = F \\cdot s \\cdot \\cos\\alpha$).",
            "- **Джоуль (Дж)** — единица энергии и работы в СИ: $1\\,\\text{Дж} = 1\\,\\text{Н} \\cdot 1\\,\\text{м} = 1\\,\\text{кг} \\cdot \\text{м}^2/\\text{с}^2$."
          ].join("\n");
        }
      }

      if (id === "phys-newton-01") {
        if (promptType === "summary") {
          return [
            "### Краткий пересказ: Законы динамики Ньютона",
            "",
            "- **I закон**: Существование инерциальных систем отсчёта (ИСО), где тело сохраняет покой или скорость при $\\sum \\vec{F} = 0$.",
            "- **II закон**: Связь причины и следствия движения: $\\vec{F} = m \\vec{a}$. Сила порождает ускорение.",
            "- **III закон**: Силы взаимодействия равны по модулю и противоположны по знаку: $\\vec{F}_1 = -\\vec{F}_2$ (приложены к разным телам)."
          ].join("\n");
        }
        if (promptType === "exam_questions") {
          return [
            "### Экзаменационные вопросы",
            "",
            "1. **Почему силы действия и противодействия не уравновешивают друг друга?**",
            "   *Ответ*: Они приложены к **разным** взаимодействующим телам, а не к одному.",
            "",
            "2. **Чему равен вес тела при свободном падении с ускорением $g$?**",
            "   *Ответ*: Вес равен нулю ($P = 0$), тело находится в состоянии полной невесомости."
          ].join("\n");
        }
        if (promptType === "glossary") {
          return [
            "### Глоссарий терминов",
            "",
            "- **Инерция** — свойство тела сохранять скорость неизменной при отсутствии внешних воздействий.",
            "- **Инертность** — свойство сопротивляться изменению скорости, количественной мерой которого является масса $m$.",
            "- **Ньютон (Н)** — единица силы: сила, сообщающая телу массой $1\\,\\text{кг}$ ускорение $1\\,\\text{м/с}^2$."
          ].join("\n");
        }
      }

      if (id === "hist-1812-01") {
        if (promptType === "summary") {
          return [
            "### Краткий пересказ: Отечественная война 1812 г.",
            "",
            "- **Причина**: Разрыв Россией разорительной Континентальной блокады Англии и гегемонистские амбиции Наполеона.",
            "- **Стратегия**: Затягивание противника вглубь территории (Барклай), генеральное сражение под Бородино и Тарутинский манёвр (Кутузов).",
            "- **Итог**: Полное уничтожение Великой армии и крах господства Франции в Европе."
          ].join("\n");
        }
        if (promptType === "exam_questions") {
          return [
            "### Вопросы к экзамену",
            "",
            "1. **Каковы главные причины вторжения Наполеона в Россию?**",
            "   *Ответ*: Несоблюдение Россией условий континентальной блокады Англии и польский вопрос.",
            "",
            "2. **В чём военно-стратегическое значение Тарутинского манёвра?**",
            "   *Ответ*: Русская армия прикрыла Тульский оружейный завод и богатые южные губернии, отрезав Наполеону путь отступления."
          ].join("\n");
        }
        if (promptType === "glossary") {
          return [
            "### Исторический глоссарий",
            "",
            "- **Континентальная блокада** — система экономических санкций против Великобритании, объявленная Наполеоном в 1806 г.",
            "- **Флеши** — полевые земляные укрепления в форме наконечника стрелы (Багратионовы флеши).",
            "- **Партизанское движение 1812 г.** — действия армейских кавалерийских летучих отрядов и крестьянского ополчения в тылу врага."
          ].join("\n");
        }
      }

      if (id === "lit-onegin-01") {
        if (promptType === "summary") {
          return [
            "### Краткий пересказ: «Евгений Онегин»",
            "",
            "- **Жанр**: Роман в стихах, новаторская форма с лирическими отступлениями автора.",
            "- **Форма**: Онегинская строфа из $14$ строк ямба со схемой $AbAb\\,CCdd\\,EffE\\,gg$.",
            "- **Идея**: Исследование типа «лишнего человека» и противопоставление светской фальши искренней нравственной чистоте Татьяны."
          ].join("\n");
        }
        if (promptType === "exam_questions") {
          return [
            "### Вопросы к экзамену",
            "",
            "1. **Почему Онегин принял вызов Ленского на дуэль, осознавая нелепость ссоры?**",
            "   *Ответ*: Из-за страха перед насмешками и сплетнями уездного светского общества (ложное понятие о чести).",
            "",
            "2. **Какова схема рифмовки онегинской строфы?**",
            "   *Ответ*: Перекрёстная ($AbAb$), смежная ($CCdd$), опоясывающая ($EffE$) и заключительное парное двустишие ($gg$)."
          ].join("\n");
        }
        if (promptType === "glossary") {
          return [
            "### Литературный глоссарий",
            "",
            "- **Лишний человек** — литературный тип дворянина, обладающего умом и талантом, но не находящего применения силам в обществе.",
            "- **Онегинская строфа** — $14$-строчная поэтическая форма четырёхстопного ямба, разработанная Пушкиным.",
            "- **Лирическое отступление** — внефабульный элемент произведения, где автор делится личными размышлениями и чувствами."
          ].join("\n");
        }
      }

      if (id === "bio-cell-01") {
        if (promptType === "summary") {
          return [
            "### Краткий пересказ: Строение клетки",
            "",
            "- **Цитология**: Изучает клетку как базовую структурно-функциональную единицу живого.",
            "- **Дихотомия**: Прокариоты (доядерные без мембранных органоидов) vs Эукариоты (истинное ядро и компартменты).",
            "- **Энергетика**: Митохондрии синтезируют АТФ в процессе окислительного фосфорилирования."
          ].join("\n");
        }
        if (promptType === "exam_questions") {
          return [
            "### Вопросы к экзамену",
            "",
            "1. **Каковы различия между прокариотической и эукариотической клеткой?**",
            "   *Ответ*: Наличие оформленного ядра с мембраной и мембранных органоидов у эукариот.",
            "",
            "2. **В чём заключается сущность жидкостно-мозаичной модели мембраны?**",
            "   *Ответ*: Липидный бислой находится в жидком агрегатном состоянии, а молекулы белков плавают в нём подобно айсбергам."
          ].join("\n");
        }
        if (promptType === "glossary") {
          return [
            "### Цитологический глоссарий",
            "",
            "- **АТФ (аденозинтрифосфат)** — универсальный аккумулятор и переносчик химической энергии в клетке.",
            "- **Рибосома** — немембранная органелла, осуществляющая трансляцию генетического кода в белковую молекулу.",
            "- **Митохондрия** — двумембранная органелла эукариотической клетки, обеспечивающая синтез АТФ."
          ].join("\n");
        }
      }

      if (id === "chem-sol-01") {
        if (promptType === "summary") {
          return [
            "### Краткий пересказ: Растворы и концентрации",
            "",
            "- **Раствор**: Однородная система из растворителя и растворённого вещества.",
            "- **Массовая доля**: Доля вещества в общей массе раствора: $\\omega = \\frac{m_{\\text{в-ва}}}{m_{\\text{р-ра}}} \\times 100\\%$.",
            "- **Молярность**: Отношение количества вещества в молях к литрам объёма: $C_M = \\frac{\\nu}{V}$."
          ].join("\n");
        }
        if (promptType === "exam_questions") {
          return [
            "### Вопросы к экзамену",
            "",
            "1. **Как изменится массовая доля соли, если к 200 г 10% раствора добавить 200 г чистой воды?**",
            "   *Ответ*: Масса раствора удвоится ($400\\,\\text{г}$), а концентрация снизится вдвое — до $5\\%$.",
            "",
            "2. **Что такое кристаллогидрат? Приведите пример.**",
            "   *Ответ*: Кристаллическое вещество, содержащее химически связанную воду, например $\\text{CuSO}_4 \\cdot 5\\text{H}_2\\text{O}$."
          ].join("\n");
        }
        if (promptType === "glossary") {
          return [
            "### Химический глоссарий",
            "",
            "- **Массовая доля ($\\omega$)** — отношение массы компонента к массе всей системы.",
            "- **Молярная концентрация ($C_M$)** — количество вещества (моль), содержащееся в $1\\,\\text{л}$ раствора.",
            "- **Насыщенный раствор** — раствор, находящийся в динамическом равновесии с нерастворённым веществом при данной температуре."
          ].join("\n");
        }
      }

      // Universal dynamic generator for custom questions and user-uploaded scans
      if (customPrompt) {
        return [
          "### Анализ вопроса: «" + escapeHtml(customPrompt) + "»",
          "",
          "На основе анализа документа **«" + (note.title || "Конспект") + "»** (" + (note.subject || "Предмет") + "):",
          "",
          "- В конспекте зафиксированы ключевые закономерности и теоретические положения по данной теме.",
          "- Рассматриваемый аспект напрямую соотносится с предметным базисом и подтверждается структурированными выводами конспекта.",
          "- *Рекомендация*: Для углублённого изучения обратите внимание на формулы и формулировки в основном тексте конспекта."
        ].join("\n");
      }

      if (promptType === "summary") {
        return [
          "### Краткий пересказ конспекта",
          "",
          "- **Тема**: " + note.title + " (" + note.subject + ").",
          "- **Степень компрессии**: " + (note.compressionRatio ? note.compressionRatio.toFixed(1) + "%" : "высокая") + " (исходный вес " + formatBytes(note.rawBytes) + " сжат до " + formatBytes(note.cleanBytes) + ").",
          "- **Содержание**: Документ успешно нормализован и очищен от шумов фонового освещения и бумаги. Распознанный текст приведён в левом окне редактора."
        ].join("\n");
      }

      if (promptType === "exam_questions") {
        return [
          "### Контрольные вопросы по конспекту",
          "",
          "1. Сформулируйте основную идею и предмет изучения темы «" + note.title + "».",
          "2. Какие фундаментальные формулы или даты являются ключевыми в данном конспекте?",
          "3. Как практические примеры из текста иллюстрируют общую теоретическую модель?"
        ].join("\n");
      }

      return [
        "### Термины и определения",
        "",
        "- **" + note.title + "** — центральное понятие представленного конспекта.",
        "- **Векторный слой** — оптимизированное представление рукописного документа без потери читаемости символов."
      ].join("\n");
    }
  }

  var geminiProcessor = new GeminiNoteProcessor();

  /* ==============================================================================
     7. READER STUDIO CONTROLLER
     ============================================================================== */
  var currentDoc = null;

  function updateGeminiStatusIndicator() {
    var pill = $("#gemini-status-pill");
    var text = $("#gemini-status-text");
    if (!pill || !text) return;
    if (geminiProcessor.hasApiKey()) {
      text.textContent = "Gemini 1.5 Flash (Онлайн)";
      pill.style.borderColor = "rgba(16, 185, 129, 0.4)";
      var dot = pill.querySelector(".status-dot");
      if (dot) {
        dot.style.background = "#10b981";
        dot.style.boxShadow = "0 0 8px #10b981";
      }
    } else {
      text.textContent = "Автономный режим (Офлайн)";
      pill.style.borderColor = "rgba(255, 255, 255, 0.12)";
      var dotOff = pill.querySelector(".status-dot");
      if (dotOff) {
        dotOff.style.background = "#eab308";
        dotOff.style.boxShadow = "0 0 6px #eab308";
      }
    }
  }

  function openReader(item) {
    if (!item) return;
    currentDoc = item;
    closeHuds();

    var screen = $("#reader-screen");
    if (!screen) return;

    var ratio = item.compressionRatio ? item.compressionRatio.toFixed(1) + "%" : "—";
    var setText = function (sel, val) { var el = $(sel); if (el) el.textContent = val; };

    setText("#reader-doc-title", item.title);
    setText("#reader-doc-badge", item.subject + " · " + item.date);
    setText("#reader-stat-ratio", ratio);
    setText("#reader-stat-size", formatBytes(item.cleanBytes || 0));
    setText("#reader-scan-meta", "RAW SCAN · " + formatBytes(item.rawBytes || 0));

    var img = $("#reader-scan-img");
    if (img) {
      img.src = item.rawUrl;
      img.alt = "Скан: " + item.title;
      img.style.transformOrigin = "center center";
    }

    var wrap = $("#scan-zoom-wrap");
    if (wrap && wrap._resetZoom) wrap._resetZoom();

    var bodyEl = $("#reader-content-body");
    if (bodyEl) {
      bodyEl.innerHTML = renderMarkdownToHtml(item.markdownBody || item.rawOcrText);
      bodyEl.scrollTop = 0;
      ensureKatexHydration(bodyEl);
    }

    // Reset Gemini drawer response state
    var respWrap = $("#gemini-response-wrap");
    if (respWrap) respWrap.hidden = true;
    var respBody = $("#gemini-response-body");
    if (respBody) respBody.innerHTML = "";
    $$(".ai-pill").forEach(function (p) { p.classList.remove("is-active"); });
    updateGeminiStatusIndicator();

    // Pause ambient video to save CPU/GPU and maximize reading focus
    var bgVideo = $("#bg-video");
    if (bgVideo && !bgVideo.paused) {
      try {
        bgVideo.pause();
        bgVideo._wasPausedByReader = true;
      } catch (e) {}
    }

    screen.hidden = false;
    document.body.setAttribute("data-view", "reader");

    var back = $("#btn-reader-back");
    if (back) back.focus();
  }

  function closeReader(reopenLibrary) {
    var screen = $("#reader-screen");
    document.body.setAttribute("data-view", "home");

    // Resume video if it was playing before
    var bgVideo = $("#bg-video");
    if (bgVideo && bgVideo._wasPausedByReader && location.protocol !== "file:") {
      try { bgVideo.play().catch(function () {}); } catch (e) {}
      bgVideo._wasPausedByReader = false;
    }

    if (screen) {
      window.setTimeout(function () {
        if (document.body.getAttribute("data-view") !== "reader") {
          screen.hidden = true;
        }
      }, 300);
    }

    if (reopenLibrary) {
      openHud("hud-library", $('a[href="#library"]'));
    }
  }

  /* ==============================================================================
     8. 2X SCAN ZOOM & PAN TRACKING CONTROLLER
     ============================================================================== */
  function bootScanZoom() {
    var wrap = $("#scan-zoom-wrap");
    var img = $("#reader-scan-img");
    var badge = $("#zoom-badge");
    var hint = $("#zoom-hint");
    if (!wrap || !img) return;

    var isZoomed = false;

    function setZoomState(zoomed) {
      isZoomed = zoomed;
      wrap.classList.toggle("is-zoomed", isZoomed);
      if (badge) badge.textContent = isZoomed ? "2.0×" : "1.0×";
      if (hint) {
        hint.textContent = isZoomed
          ? "Перемещайте мышь для обзора · клик для выхода"
          : "Кликните для 2× зума";
      }
      if (!isZoomed) {
        img.style.transformOrigin = "center center";
      }
    }

    wrap.addEventListener("click", function (e) {
      setZoomState(!isZoomed);
      if (isZoomed) {
        var rect = wrap.getBoundingClientRect();
        var x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
        var y = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));
        img.style.transformOrigin = x.toFixed(1) + "% " + y.toFixed(1) + "%";
      }
    });

    wrap.addEventListener("pointermove", function (e) {
      if (!isZoomed) return;
      var rect = wrap.getBoundingClientRect();
      var x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
      var y = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));
      img.style.transformOrigin = x.toFixed(1) + "% " + y.toFixed(1) + "%";
    });

    wrap.addEventListener("pointerleave", function () {
      if (!isZoomed) {
        img.style.transformOrigin = "center center";
      }
    });

    wrap._resetZoom = function () {
      setZoomState(false);
    };
  }

  /* ==============================================================================
     9. GEMINI DRAWER & STREAMING TYPEWRITER CONTROLLER
     ============================================================================== */
  var currentTypewriterCancel = null;

  function streamTextToElement(element, rawText, onComplete) {
    if (currentTypewriterCancel) currentTypewriterCancel();
    element.innerHTML = '<span class="cursor-pulse"></span>';

    var i = 0;
    var speed = 10;
    var stepSize = Math.max(1, Math.floor(rawText.length / 110));
    var timer = null;

    function tick() {
      i += stepSize;
      if (i >= rawText.length) {
        i = rawText.length;
        window.clearInterval(timer);
        currentTypewriterCancel = null;
        element.innerHTML = renderMarkdownToHtml(rawText);
        ensureKatexHydration(element);
        if (onComplete) onComplete();
        return;
      }
      var slice = rawText.slice(0, i);
      element.innerHTML = renderMarkdownToHtml(slice) + '<span class="cursor-pulse"></span>';
    }

    timer = window.setInterval(tick, speed);

    currentTypewriterCancel = function () {
      window.clearInterval(timer);
      currentTypewriterCancel = null;
      element.innerHTML = renderMarkdownToHtml(rawText);
      ensureKatexHydration(element);
    };
  }

  function triggerGeminiAction(promptType, customQuery) {
    if (!currentDoc) return;
    var respWrap = $("#gemini-response-wrap");
    var respLabel = $("#gemini-response-label");
    var respBody = $("#gemini-response-body");
    if (!respWrap || !respBody) return;

    respWrap.hidden = false;

    // Update label
    var labels = {
      summary: "Краткий пересказ · Gemini 1.5",
      exam_questions: "Вопросы к экзамену · Gemini 1.5",
      glossary: "Разбор ключевых терминов · Gemini 1.5"
    };
    if (respLabel) {
      respLabel.textContent = labels[promptType] || "Ответ ассистента · Gemini 1.5";
    }

    // Active pill styling
    $$(".ai-pill").forEach(function (pill) {
      var match = pill.getAttribute("data-type") === promptType;
      pill.classList.toggle("is-active", match);
    });

    respBody.innerHTML = '<span class="cursor-pulse"></span>';

    geminiProcessor.processNote(currentDoc, promptType, customQuery).then(function (res) {
      streamTextToElement(respBody, res.text, function () {
        // finished streaming
      });
    }).catch(function (err) {
      respBody.innerHTML = "<p>Ошибка обработки: " + escapeHtml(err.message) + "</p>";
    });
  }

  function bootGeminiDrawer() {
    // Action pills
    $$(".ai-pill").forEach(function (pill) {
      pill.addEventListener("click", function () {
        var pType = pill.getAttribute("data-type");
        triggerGeminiAction(pType, null);
      });
    });

    // Custom query input
    var submitBtn = $("#btn-gemini-submit");
    var customInput = $("#gemini-custom-input");

    function handleCustomSubmit() {
      if (!customInput) return;
      var q = customInput.value.trim();
      if (!q) return;
      customInput.value = "";
      triggerGeminiAction("custom", q);
    }

    if (submitBtn) submitBtn.addEventListener("click", handleCustomSubmit);
    if (customInput) {
      customInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
          e.preventDefault();
          handleCustomSubmit();
        }
      });
    }

    // Copy Gemini response
    var copyBtn = $("#btn-copy-gemini");
    if (copyBtn) {
      copyBtn.addEventListener("click", async function () {
        var respBody = $("#gemini-response-body");
        if (!respBody) return;
        var text = respBody.innerText || respBody.textContent || "";
        try {
          await navigator.clipboard.writeText(text);
          showToast("Ответ Gemini скопирован!");
        } catch (e) {
          showToast("Скопировано!");
        }
      });
    }

    // Gemini API Key Modal
    var keyModalBtn = $("#btn-open-key-modal");
    if (keyModalBtn) {
      keyModalBtn.addEventListener("click", function () {
        var input = $("#gemini-key-input");
        if (input) input.value = geminiProcessor.getApiKey();
        openHud("hud-gemini-key", null);
      });
    }

    var saveKeyBtn = $("#btn-save-gemini-key");
    if (saveKeyBtn) {
      saveKeyBtn.addEventListener("click", function () {
        var input = $("#gemini-key-input");
        var val = input ? input.value : "";
        geminiProcessor.setApiKey(val);
        updateGeminiStatusIndicator();
        closeHuds();
        showToast(val ? "Ключ Gemini сохранён" : "Ключ очищен, включён офлайн-режим");
      });
    }
  }

  /* ==============================================================================
     10. LIBRARY TABLE & HUD CONTROLLER
     ============================================================================== */
  function renderLibraryTable() {
    var tbody = $("#library-tbody");
    if (!tbody) return;
    var allItems = getAllNotes();

    tbody.innerHTML = allItems.map(function (item) {
      var badge = item.isUserUploaded ? '<span class="doc-badge">Пользовательский</span>' : '<span class="doc-badge">Демо</span>';
      var pillClass = item.isUserUploaded ? "status-pill is-user" : "status-pill";
      var thumb = item.thumb || item.rawUrl ? '<img class="doc-thumb" src="' + escapeHtml(item.thumb || item.rawUrl) + '" alt="">' : "";
      var ratioStr = item.compressionRatio ? item.compressionRatio.toFixed(1) + "%" : "—";
      return (
        '<tr data-id="' + escapeHtml(item.id) + '">' +
          '<td><div class="doc-name">' + thumb + '<span>' + escapeHtml(item.title) + "</span>" + badge + "</div></td>" +
          "<td>" + escapeHtml(item.subject) + "</td>" +
          "<td>" + escapeHtml(item.date) + "</td>" +
          "<td>" + formatBytes(item.cleanBytes || item.rawBytes) + "</td>" +
          "<td>" + ratioStr + "</td>" +
          '<td><span class="' + pillClass + '">' + (item.isUserUploaded ? "Свой" : "Вектор") + "</span></td>" +
          '<td style="text-align:right;"><button class="row-action" type="button" data-open="' + escapeHtml(item.id) + '">Открыть</button></td>' +
        "</tr>"
      );
    }).join("");
  }

  function closeHuds() {
    $$(".hud").forEach(function (h) { h.classList.remove("is-open"); });
    $$(".nav-pill").forEach(function (a) { a.classList.remove("is-active"); });
    var scrim = $("#hud-scrim");
    if (scrim) scrim.classList.remove("is-open");
  }

  function openHud(id, pill) {
    var hud = document.getElementById(id);
    var already = hud && hud.classList.contains("is-open");
    closeHuds();
    if (!already && hud) {
      hud.classList.add("is-open");
      var scrim = $("#hud-scrim");
      if (scrim) scrim.classList.add("is-open");
      if (pill) pill.classList.add("is-active");
    }
  }

  /* ==============================================================================
     11. LANDING LOUPE & IMAGE INGESTION PIPELINE
     ============================================================================== */
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

  var objectUrls = [];
  function revokeLater(url) {
    objectUrls.push(url);
    if (objectUrls.length > 8) {
      var old = objectUrls.shift();
      try { URL.revokeObjectURL(old); } catch (e) {}
    }
  }

  function canvasToUrl(canvas, type, quality) {
    return new Promise(function (resolve) {
      canvas.toBlob(function (blob) {
        if (!blob) {
          resolve(canvas.toDataURL(type || "image/jpeg", quality || 0.85));
          return;
        }
        var url = URL.createObjectURL(blob);
        revokeLater(url);
        resolve(url);
      }, type || "image/jpeg", quality || 0.85);
    });
  }

  function drawOverlay(preCanvas, text, extra) {
    var w = preCanvas.width;
    var h = preCanvas.height;
    var out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    var ctx = out.getContext("2d");
    ctx.fillStyle = "#f6f6f4";
    ctx.fillRect(0, 0, w, h);
    ctx.globalAlpha = 0.22;
    ctx.drawImage(preCanvas, 0, 0);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, 4, h);
    ctx.fillStyle = "#8a8a8a";
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.fillText("GLYPH · VECTOR OUTPUT", 22, 28);
    ctx.fillStyle = "#141414";
    ctx.font = "600 22px Inter, system-ui, sans-serif";
    ctx.fillText(extra && extra.title ? extra.title : "Распознанный конспект", 22, 58);
    ctx.strokeStyle = "#e1e1e1";
    ctx.beginPath();
    ctx.moveTo(22, 72);
    ctx.lineTo(w - 22, 72);
    ctx.stroke();

    var lines = String(text || "").split("\n");
    var y = 100;
    ctx.fillStyle = "#1c1c1c";
    ctx.font = "15px Inter, system-ui, sans-serif";
    var used = 0;
    for (var i = 0; i < lines.length && y < h - 90; i++) {
      var line = lines[i].replace(/\s+/g, " ").trim();
      if (!line) { y += 12; continue; }
      if (line.length > 60) line = line.slice(0, 60) + "…";
      ctx.fillText(line, 22, y);
      y += 22;
      used++;
      if (used > 20) break;
    }
    ctx.fillStyle = "#8a8a8a";
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.fillText("GLYPH · VECTOR SHEET", 22, h - 24);
    return out;
  }

  function updateTelemetry(orig, comp) {
    var cs = $("#chip-size");
    if (cs) cs.textContent = "[ СЖАТИЕ: " + formatBytes(orig) + " → " + formatBytes(comp) + " ]";
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

  async function processUploadedImage(img, fileMeta) {
    var loupe = $("#loupe");
    if (loupe) loupe.classList.add("is-scanning");
    setStatus("SCAN · NORMALIZE");
    setSplit(8);

    var rawCanvas = GlyphEngine.drawScaled(img);
    var rawUrl = await canvasToUrl(rawCanvas, "image/jpeg", 0.86);
    setLayer($("#layer-raw"), rawUrl);

    setStatus("SCAN · CONTRAST");
    var pre = GlyphEngine.preprocessCanvas(rawCanvas);
    var compressed = await GlyphEngine.compressOriginal(img, fileMeta.bytes || 0);
    var compressedBytes = compressed && compressed.bytes ? compressed.bytes : Math.round((fileMeta.bytes || 1) * 0.05);
    updateTelemetry(fileMeta.bytes || rawCanvas.width * rawCanvas.height * 3, compressedBytes);

    var ocrText = "";
    try {
      setStatus("SCAN · VECTOR");
      var rec = await GlyphEngine.recognize(pre, function (m) {
        if (m.status === "recognizing text") {
          setStatus("SCAN · VECTOR " + Math.round((m.progress || 0) * 100) + "%");
        }
      });
      ocrText = rec.text || rec.raw || "";
    } catch (err) {
      ocrText = fileMeta.fallback || "Конспект нормализован. Векторный бинарный слой подготовлен.";
    }

    var overlay = drawOverlay(pre, ocrText, { title: fileMeta.title || "Векторный слой" });
    var cleanUrl = await canvasToUrl(overlay, "image/jpeg", 0.9);
    setLayer($("#layer-clean"), cleanUrl);

    await animateSplit(8, 88, 1400);
    if (loupe) loupe.classList.remove("is-scanning");
    setStatus("SPECTRAL 1200 DPI");

    return {
      rawUrl: rawUrl,
      cleanUrl: cleanUrl,
      cleanBytes: compressedBytes,
      text: ocrText
    };
  }

  async function ingestFile(file) {
    if (!file || !file.type || file.type.indexOf("image") !== 0) {
      setStatus("НУЖЕН СНИМОК");
      return;
    }
    var url = GlyphEngine.fileToUrl(file);
    revokeLater(url);
    var img = await GlyphEngine.loadImage(url);
    var title = file.name.replace(/\.[^.]+$/, "") || "Новый конспект";
    var result = await processUploadedImage(img, { bytes: file.size, title: title });
    var ratio = file.size ? Math.max(0, (1 - result.cleanBytes / file.size) * 100) : 0;

    var today = new Date();
    var dd = String(today.getDate()).padStart(2, "0");
    var mm = String(today.getMonth() + 1).padStart(2, "0");
    var dateStr = dd + "." + mm + "." + today.getFullYear();

    var markdownBody = [
      "# " + title,
      "",
      "> Пользовательский скан · " + dateStr + " · Сжатие " + ratio.toFixed(1) + "%",
      "",
      "## Распознанный текст",
      result.text || "Текст конспекта успешно оцифрован."
    ].join("\n");

    var newNote = {
      id: "user-" + Date.now(),
      title: title,
      subject: "Пользовательский скан",
      date: dateStr,
      rawUrl: result.rawUrl || url,
      cleanUrl: result.cleanUrl,
      thumb: result.rawUrl || url,
      rawBytes: file.size,
      cleanBytes: result.cleanBytes,
      compressionRatio: ratio,
      rawOcrText: result.text,
      markdownBody: markdownBody,
      isUserUploaded: true
    };

    saveUserNote(newNote);
    renderLibraryTable();
    showToast("Конспект добавлен в библиотеку!");
  }

  function openPicker() {
    var input = $("#file-input");
    if (input) input.click();
  }

  function bootDropzone() {
    var stage = $("#loupe-stage");
    var input = $("#file-input");
    if (!stage || !input) return;

    ["dragenter", "dragover"].forEach(function (ev) {
      stage.addEventListener(ev, function (e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        stage.classList.add("is-over");
      });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      stage.addEventListener(ev, function (e) {
        e.preventDefault();
        if (ev === "drop") {
          var f = e.dataTransfer.files && e.dataTransfer.files[0];
          if (f) ingestFile(f);
        }
        stage.classList.remove("is-over");
      });
    });
    stage.addEventListener("click", function (e) {
      if (e.target.closest(".splitter") || e.target.closest(".handle")) return;
      openPicker();
    });
    stage.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(); }
    });
    input.addEventListener("change", function () {
      if (input.files && input.files[0]) ingestFile(input.files[0]);
    });
    var btn = $("#btn-upload");
    if (btn) btn.addEventListener("click", openPicker);
    var libUp = $("#btn-lib-upload");
    if (libUp) libUp.addEventListener("click", function (e) {
      e.stopPropagation();
      openPicker();
    });
  }

  function bootSplitter() {
    var stage = $("#loupe-stage");
    var loupe = $("#loupe");
    if (!stage || !loupe) return;
    var dragging = false;
    function pos(ev) {
      var rect = stage.getBoundingClientRect();
      var x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
      setSplit((x / rect.width) * 100);
    }
    stage.addEventListener("pointerdown", function (e) {
      if (e.target.closest(".handle") || e.target.closest(".splitter") || e.offsetX > stage.clientWidth * 0.02) {
        dragging = true;
        loupe.classList.add("is-dragging");
        try { stage.setPointerCapture(e.pointerId); } catch (err) {}
        pos(e);
      }
    });
    stage.addEventListener("pointermove", function (e) { if (dragging) pos(e); });
    stage.addEventListener("pointerup", function () { dragging = false; loupe.classList.remove("is-dragging"); });
    stage.addEventListener("pointercancel", function () { dragging = false; loupe.classList.remove("is-dragging"); });
  }

  function bootParallax() {
    var col = $("#loupe-col");
    var loupe = $("#loupe");
    var glare = $("#loupe-glare");
    if (!col || !loupe) return;
    if (window.matchMedia("(max-width: 900px)").matches) return;
    col.addEventListener("pointermove", function (e) {
      var r = loupe.getBoundingClientRect();
      var px = (e.clientX - r.left) / r.width - 0.5;
      var py = (e.clientY - r.top) / r.height - 0.5;
      var rx = (py * -4).toFixed(2);
      var ry = (px * 4).toFixed(2);
      loupe.style.transform = "rotateX(" + rx + "deg) rotateY(" + ry + "deg)";
      if (glare) {
        glare.style.setProperty("--gx", (50 + px * 40) + "%");
        glare.style.setProperty("--gy", (30 + py * 30) + "%");
      }
    });
    col.addEventListener("pointerleave", function () {
      loupe.style.transform = "rotateX(0deg) rotateY(0deg)";
    });
  }

  /* ==============================================================================
     12. NAVIGATION & READER BOOTSTRAP
     ============================================================================== */
  function bootReader() {
    var back = $("#btn-reader-back");
    if (back) back.addEventListener("click", function () { closeReader(true); });

    var copyMd = $("#btn-copy-md");
    if (copyMd) {
      copyMd.addEventListener("click", async function () {
        if (!currentDoc) return;
        var md = currentDoc.markdownBody || currentDoc.rawOcrText;
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
    }

    var dlTex = $("#btn-download-tex");
    if (dlTex) {
      dlTex.addEventListener("click", function () {
        if (!currentDoc) return;
        var texContent = noteToLatex(currentDoc);
        var blob = new Blob([texContent], { type: "application/x-tex" });
        var filename = (currentDoc.id || "glyph-note") + ".tex";
        if (window.GlyphEngine && GlyphEngine.downloadBlob) {
          GlyphEngine.downloadBlob(blob, filename);
        }
        showToast("LaTeX сохранён");
      });
    }

    var tabs = { scan: $("#tab-scan"), text: $("#tab-text") };
    var ws = $("#reader-workspace");
    Object.keys(tabs).forEach(function (key) {
      var btn = tabs[key];
      if (!btn || !ws) return;
      btn.addEventListener("click", function () {
        ws.setAttribute("data-pane", key);
        Object.keys(tabs).forEach(function (k) {
          if (!tabs[k]) return;
          var active = k === key;
          tabs[k].classList.toggle("is-active", active);
          tabs[k].setAttribute("aria-selected", active ? "true" : "false");
        });
      });
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && document.body.getAttribute("data-view") === "reader") {
        closeReader(false);
      }
    });

    bootScanZoom();
    bootGeminiDrawer();
  }

  function bootNav() {
    var burger = $("#burger-btn");
    var backdrop = $("#menu-backdrop");
    function setMenu(open) {
      document.body.classList.toggle("menu-open", open);
      if (burger) {
        burger.setAttribute("aria-expanded", open ? "true" : "false");
        burger.setAttribute("aria-label", open ? "Закрыть меню" : "Открыть меню");
      }
    }
    if (burger) burger.addEventListener("click", function () {
      setMenu(!document.body.classList.contains("menu-open"));
    });
    if (backdrop) backdrop.addEventListener("click", function () { setMenu(false); closeHuds(); });
    var scrim = $("#hud-scrim");
    if (scrim) scrim.addEventListener("click", closeHuds);
    $$(".hud-close").forEach(function (b) { b.addEventListener("click", closeHuds); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && document.body.getAttribute("data-view") !== "reader") {
        setMenu(false);
        closeHuds();
      }
    });
    window.addEventListener("resize", function () {
      if (window.innerWidth > 900) setMenu(false);
    });

    $$(".nav-pill").forEach(function (a) {
      a.addEventListener("click", function (e) {
        var href = a.getAttribute("href") || "";
        if (href === "#library" || href === "#engine" || href === "#architecture" || href === "#benchmarks") {
          e.preventDefault();
          openHud("hud-" + href.slice(1), a);
          setMenu(false);
        }
      });
    });

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
        var all = getAllNotes();
        var item = all.filter(function (it) { return it.id === id; })[0];
        if (item) openReader(item);
      });
    }
  }

  /* ==============================================================================
     13. AMBIENT BACKGROUND CANVAS FALLBACK & VIDEO CONTROLLER
     ============================================================================== */
  function initAmbientCanvas(canvas) {
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var particles = [];
    var w = 0, h = 0, t = 0, raf;

    function resize() {
      w = window.innerWidth; h = window.innerHeight;
      canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + "px"; canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      particles = [];
      var n = Math.floor((w * h) / 14000);
      for (var i = 0; i < n; i++) {
        particles.push({
          x: Math.random() * w,
          y: Math.random() * h,
          r: Math.random() * 1.6 + 0.3,
          s: Math.random() * 0.35 + 0.08,
          a: Math.random() * 0.35 + 0.08
        });
      }
    }

    function tick() {
      t += 0.008;
      ctx.fillStyle = "rgba(0,0,0,0.22)";
      ctx.fillRect(0, 0, w, h);
      var g = ctx.createRadialGradient(w * 0.5, h * 0.62, 40, w * 0.5, h * 0.55, Math.max(w, h) * 0.7);
      g.addColorStop(0, "rgba(70,70,74,0.16)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
      for (var i = 0; i < particles.length; i++) {
        var p = particles[i];
        p.x += Math.sin(t + p.y * 0.01) * p.s;
        p.y -= p.s * 0.45;
        if (p.y < -8) { p.y = h + 8; p.x = Math.random() * w; }
        ctx.beginPath();
        ctx.fillStyle = "rgba(230,230,230," + p.a + ")";
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      raf = window.requestAnimationFrame(tick);
    }

    resize();
    window.addEventListener("resize", resize);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    tick();
  }

  function bootVideo() {
    var video = $("#bg-video");
    var canvas = $("#ambient-canvas");
    var shell = $("#video-container");
    if (!video || !shell) return;
    var failed = false;

    function useCanvas() {
      if (failed) return;
      failed = true;
      shell.classList.add("is-canvas");
      try { video.pause(); } catch (e) {}
      initAmbientCanvas(canvas);
    }

    if (location.protocol === "file:") {
      useCanvas();
      return;
    }

    function tryPlay() {
      video.muted = true;
      var p = video.play();
      if (p && p.catch) p.catch(function () {
        video.muted = true;
        video.play().catch(useCanvas);
      });
    }

    video.addEventListener("error", useCanvas);
    video.addEventListener("stalled", function () {
      window.setTimeout(function () { if (video.readyState < 2) useCanvas(); }, 1800);
    });
    video.addEventListener("canplay", tryPlay);
    video.addEventListener("loadeddata", tryPlay);
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden && !failed && document.body.getAttribute("data-view") !== "reader") {
        tryPlay();
      }
    });
    tryPlay();
    window.setTimeout(function () { if (video.paused || video.readyState < 2) useCanvas(); }, 2500);
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

  /* ==============================================================================
     14. DOM READY INITIALIZATION
     ============================================================================== */
  document.addEventListener("DOMContentLoaded", function () {
    reveal();
    bootVideo();
    bootSplitter();
    bootParallax();
    bootNav();
    bootDropzone();
    bootReader();
    renderLibraryTable();
    updateGeminiStatusIndicator();
    document.body.setAttribute("data-view", "home");
    setSplit(42);

    if (window.GlyphEngine && GlyphEngine.ensureWorker) {
      GlyphEngine.ensureWorker(function () {}).catch(function () {});
    }
  });
})();
