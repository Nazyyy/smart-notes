#!/usr/bin/env bash
# ### FILE: scripts/run_local.sh
# ==============================================================================
# SMART NOTES - LOCAL DEVELOPMENT RUNNER
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Activate virtual environment if present
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# For local development outside Docker, default to SQLite if postgres host is not available
if [ -z "${DATABASE_URL:-}" ] || [[ "${DATABASE_URL:-}" == *"@postgres:"* ]]; then
    mkdir -p "$ROOT_DIR/data"
    export DATABASE_URL="sqlite+aiosqlite:///$ROOT_DIR/data/smart_notes_local.db"
fi

export BACKEND_API_URL="http://127.0.0.1:8000/api/v1"

PYTHON_BIN="python3"
UVICORN_BIN="uvicorn"
STREAMLIT_BIN="streamlit"

if [ -f "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
if [ -f "$ROOT_DIR/.venv/bin/uvicorn" ]; then
    UVICORN_BIN="$ROOT_DIR/.venv/bin/uvicorn"
fi
if [ -f "$ROOT_DIR/.venv/bin/streamlit" ]; then
    STREAMLIT_BIN="$ROOT_DIR/.venv/bin/streamlit"
fi

# Terminate any stale processes on port 8000
fuser -k 8000/tcp 2>/dev/null || true

echo "=== [1/3] Verifying and Initializing Model Weights ==="
"$PYTHON_BIN" scripts/generate_synthetic_weights.py

echo "=== [2/3] Starting FastAPI Backend on http://127.0.0.1:8000 ==="
"$UVICORN_BIN" app.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

cleanup() {
    echo "Shutting down background processes..."
    kill $BACKEND_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for FastAPI Backend to be ready..."
for i in $(seq 1 40); do
    if curl -s http://127.0.0.1:8000/health | grep -q "healthy"; then
        echo "🟢 Backend is ready and healthy!"
        break
    fi
    sleep 0.5
done

echo "=== [3/3] Starting Streamlit UI on http://127.0.0.1:8501 ==="
"$STREAMLIT_BIN" run frontend/app.py --server.port 8501 --server.address 0.0.0.0
