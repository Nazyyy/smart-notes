# Smart Notes (Интеллектуальная система «Умный конспект»)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8.svg)](https://opencv.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

«Умный конспект» — это комплексная распределенная программная система корпоративного уровня для автоматизированного распознавания, очистки от оптических искажений, сегментации рукописных строк и семантического структурирования рукописных конспектов, лекций и математических формул в формат Markdown / LaTeX / TXT.

---

## 1. Архитектурный чертеж C4 (Container Diagram)

```mermaid
C4Container
    title C4 Container Diagram - Интеллектуальная система «Умный конспект»

    Person(user, "Студент / Исследователь", "Загружает фото страниц конспектов, инспектирует этапы CV, редактирует и экспортирует Markdown")

    System_Boundary(c1, "Smart Notes Platform") {
        Container(frontend, "Streamlit Frontend UI", "Python 3.11, Streamlit, Altair", "Интерактивный веб-интерфейс: drag-and-drop загрузка, визуализация стадий очистки OpenCV, интерактивный редактор строк, KaTeX превью")
        Container(backend, "FastAPI Backend Engine", "Python 3.11, FastAPI, Pydantic v2, Uvicorn", "Асинхронный REST API шлюз, потоковая валидация файлов, планировщик фоновых задач, Clean Architecture сервисы")
        Container(cv_engine, "CV Preprocessing Subsystem", "OpenCV 4.9, NumPy", "Коррекция перспективы, удаление теней, выравнивание освещения, адаптивная бинаризация, горизонтальные проекционные профили")
        Container(ml_engine, "PyTorch CRNN Inference Engine", "PyTorch 2.2, CTC Loss / Decoder", "Глубокая сверточно-рекуррентная нейросеть (CNN + BiGRU + CTC) для многоязычного HTR (кириллица, латиница, формулы)")
        ContainerDb(database, "Relational Database", "PostgreSQL 16", "Хранение метаданных документов, страниц, геометрии строк, текстов и логов обработки")
        Container(storage, "Artifact & Media Storage", "POSIX File System / S3 API compatible", "Хранение исходных изображений, промежуточных артефактов очистки и экспортированных документов")
    }

    Rel(user, frontend, "Взаимодействует через браузер", "HTTPS / Port 8501")
    Rel(frontend, backend, "Вызывает REST API эндпоинты", "HTTP/JSON / Port 8000")
    Rel(backend, database, "Выполняет транзакционные запросы (asyncpg)", "TCP / Port 5432")
    Rel(backend, storage, "Читает и записывает бинарные файлы", "File I/O / S3 API")
    Rel(backend, cv_engine, "Запускает пайплайн сегментации и очистки", "In-Process Async Executor")
    Rel(backend, ml_engine, "Передает нарезанные тензоры строк на инференс", "Batch Tensor Evaluation")
```

---

## 2. Диаграмма сущностей базы данных (ERD)

```mermaid
erDiagram
    USERS ||--o{ DOCUMENTS : owns
    DOCUMENTS ||--o{ PAGES : contains
    PAGES ||--o{ TEXT_LINES : contains
    DOCUMENTS ||--o{ EXPORT_DOCUMENTS : generates

    USERS {
        uuid id PK "Первичный ключ"
        string username UK "Уникальное имя пользователя"
        string email UK "Адрес электронной почты"
        string hashed_password "Хеш пароля Argon2/BCrypt"
        boolean is_active "Флаг активности учетной записи"
        timestamp created_at "Время создания"
        timestamp updated_at "Время последнего обновления"
    }

    DOCUMENTS {
        uuid id PK "Первичный ключ документа"
        uuid user_id FK "Владелец документа"
        string title "Название конспекта / лекции"
        string description "Краткое описание"
        string original_filename "Имя загруженного файла"
        string file_path "Путь к исходному файлу в хранилище"
        string status "PENDING | PROCESSING | COMPLETED | FAILED"
        string error_message "Текст ошибки при сбое"
        timestamp created_at "Метка времени создания"
        timestamp updated_at "Метка времени обновления"
    }

    PAGES {
        uuid id PK "Первичный ключ страницы"
        uuid document_id FK "Идентификатор документа"
        integer page_number "Порядковый номер страницы в документе"
        string raw_image_path "Путь к исходному изображению"
        string processed_image_path "Путь к бинаризованному изображению"
        string deskewed_image_path "Путь к выровненному изображению"
        string debug_dir_path "Каталог визуализаций отладки"
        integer width "Ширина изображения в пикселях"
        integer height "Высота изображения в пикселях"
        float skew_angle "Вычисленный угол наклона текста"
        string status "PENDING | PREPROCESSED | SEGMENTED | COMPLETED | FAILED"
        timestamp created_at "Метка времени создания"
    }

    TEXT_LINES {
        uuid id PK "Первичный ключ строки"
        uuid page_id FK "Идентификатор страницы"
        integer line_index "Индекс строки сверху вниз (0..N-1)"
        integer bbox_x "Координата X левого верхнего угла"
        integer bbox_y "Координата Y левого верхнего угла"
        integer bbox_w "Ширина ограничивающего прямоугольника"
        integer bbox_h "Высота ограничивающего прямоугольника"
        string recognized_text "Распознанный текст строки"
        float confidence "Оценка уверенности декодера (0.0..1.0)"
        boolean is_header "Флаг заголовка (вычисляется эвристикой)"
        boolean is_math "Флаг математического выражения"
        integer indent_level "Уровень отступа (0, 1, 2...)"
        timestamp created_at "Метка времени распознавания"
    }

    EXPORT_DOCUMENTS {
        uuid id PK "Первичный ключ экспорта"
        uuid document_id FK "Идентификатор документа"
        string export_format "MARKDOWN | TXT | LATEX | JSON"
        string file_path "Путь к сгенерированному файлу"
        text content "Полное текстовое содержимое"
        timestamp created_at "Метка времени генерации"
    }
```

---

## 3. Матрица угроз и стратегии защиты (STRIDE / OWASP)

| Угроза / Вектор атаки | Категория | Риск | Архитектурное решение / Механизм защиты |
|---|---|---|---|
| **DoS / OOM через загрузку гигантских файлов** | Denial of Service | Высокий | Потоковая буферизация `StreamingUpload` в FastAPI с ограничением максимального размера (по умолчанию 25 МБ). Принудительное ограничение разрешения (макс. 4096px по длинной стороне) с пропорциональным ресайзом в памяти. |
| **Декомпрессионная бомба (Image Decompression Bomb)** | Denial of Service | Высокий | Защита на уровне PIL / OpenCV: проверка заголовков файла, лимит пикселей `Image.MAX_IMAGE_PIXELS = 64_000_000` и перехват исключений поврежденных буферов. |
| **Инъекции в SQL-запросы** | Tampering / Information Disclosure | Критический | Применение SQLAlchemy 2.0 ORM и параметризованных запросов с типизированными моделями. Отсутствие динамической интерполяции сырых строк в SQL. |
| **Path Traversal / Local File Inclusion** | Elevation of Privilege | Критический | Генерация случайных UUIDv4 имен файлов при сохранении на диск. Изоляция хранилища в каталоге `STORAGE_DIR`, строгая санитизация путей через `pathlib.Path.resolve()` и проверка выхода за пределы корня хранилища. |
| **Переполнение памяти видеокарты / RAM при инференсе** | Denial of Service | Высокий | Батчирование нарезанных строк с фиксированным `batch_size`, автоматическое переключение `torch.no_grad()`, очистка кеша PyTorch `torch.cuda.empty_cache()` в блоках `finally`. |
| **Некорректная бинаризация при неравномерном освещении** | Fault Tolerance | Средний | Двухэтапная нормализация: вычисление фона морфологическим закрытием/открытием, деление яркостного канала на оценку фона, адаптивный порог Гаусса с локальным окном. |

---

## 4. Быстрый запуск системы

### Вариант 1: Развертывание в Docker (Рекомендуемый)

```bash
# Клонирование и переход в каталог проекта
cd /home/dima/Ai\ detector

# Запуск всего стека одной командой
docker compose up --build -d

# Проверка логов бэкенда и базы данных
docker compose logs -f backend
```

- **Веб-интерфейс Streamlit**: http://localhost:8501
- **REST API документация Swagger**: http://localhost:8000/docs
- **ReDoc API спецификация**: http://localhost:8000/redoc

### Вариант 2: Локальный запуск для разработки

```bash
# 1. Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Инициализация весов и базы данных
python3 scripts/generate_synthetic_weights.py

# 4. Запуск FastAPI бэкенда
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Запуск Streamlit фронтенда (в отдельном терминале)
streamlit run frontend/app.py --server.port 8501
```

---

## 5. Тестирование

```bash
# Запуск полного набора модульных и интеграционных тестов
pytest tests/ -v
```
