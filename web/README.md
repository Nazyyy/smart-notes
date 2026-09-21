# Glyph.ai

Один экран. Тёмный. Жидкое стекло. Клиентский пайплайн кириллических конспектов.

## Запуск

Не используй `python3 -m http.server` — у него нет HTTP 206, Chrome рвёт MP4. Подними Range-сервер:

```bash
python3 tools/serve.py 8765
```

Открыть: http://127.0.0.1:8765/

## Медиа-слой

Оригинал с CloudFront — HEVC 10-bit, Chrome на Linux его не декодирует. В фоне:

1. `assets/hero-baseline.mp4` — H.264 Constrained Baseline, yuv420p, faststart
2. `assets/hero.webm` — VP8
3. если кодек/буфер падает — watchdog глушит video и включает particle canvas

## Интерфейс

Один 100vh viewport. Никаких вкладок и плиток. Студия — жидкая лупа со split-scan на герое. Engine / Architecture / Benchmarks — микро-HUD из металла, не отдельные страницы.
