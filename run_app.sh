#!/usr/bin/env bash
# Uso: ./run_app.sh          → inicia tudo
#      ./run_app.sh --stop   → encerra tudo
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS="$DIR/.run_app.pids"

# ── stop ─────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
  [[ -f "$PIDS" ]] && while read -r p; do kill "$p" 2>/dev/null && echo "Encerrado PID $p"; done < "$PIDS" && rm -f "$PIDS"
  echo "Pronto."
  exit 0
fi

mkdir -p "$DIR/logs"

# ── .venv e dependências ──────────────────────────────────────────────────────
[[ ! -f "$DIR/.venv/bin/activate" ]] && python3.11 -m venv "$DIR/.venv"
source "$DIR/.venv/bin/activate"
python -c "import fastapi, streamlit, uvicorn" &>/dev/null \
  || pip install -r "$DIR/requirements.txt" -q

# ── Ollama ────────────────────────────────────────────────────────────────────
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
  ollama serve > "$DIR/logs/ollama.log" 2>&1 &
  echo $! >> "$PIDS"
  echo "Aguardando Ollama..."
  for i in $(seq 1 15); do
    curl -sf http://localhost:11434/api/tags &>/dev/null && break
    sleep 1
    [[ $i -eq 15 ]] && echo "Ollama não respondeu. Ver logs/ollama.log" && exit 1
  done
fi
echo "✅ Ollama   → http://localhost:11434"

# ── FastAPI ───────────────────────────────────────────────────────────────────
if ! lsof -ti tcp:8000 &>/dev/null; then
  cd "$DIR"
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > "$DIR/logs/api.log" 2>&1 &
  echo $! >> "$PIDS"
  for i in $(seq 1 10); do
    curl -sf http://localhost:8000/health &>/dev/null && break
    sleep 1
    [[ $i -eq 10 ]] && echo "API não respondeu. Ver logs/api.log" && exit 1
  done
fi
echo "✅ API      → http://localhost:8000  (docs: /docs)"

# Exibe logs da API em tempo real no terminal
tail -f "$DIR/logs/api.log" &
echo $! >> "$PIDS"

# ── Streamlit ─────────────────────────────────────────────────────────────────
cd "$DIR"
streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.headless false \
  --browser.gatherUsageStats false \
  > "$DIR/logs/streamlit.log" 2>&1 &
echo $! >> "$PIDS"
echo "✅ Streamlit → http://localhost:8501"

echo ""
echo "Para encerrar: ./run_app.sh --stop"

trap './run_app.sh --stop' INT TERM
wait
