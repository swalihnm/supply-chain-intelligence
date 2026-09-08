#!/usr/bin/env sh
# Single image, two roles. PaaS platforms inject $PORT, so honour it.
set -e

ROLE="${1:-dashboard}"

case "$ROLE" in
  dashboard)
    PORT="${PORT:-8501}"
    exec streamlit run app/dashboard.py \
      --server.port "$PORT" \
      --server.address 0.0.0.0 \
      --server.headless true \
      --browser.gatherUsageStats false
    ;;
  api)
    PORT="${PORT:-8000}"
    exec uvicorn app.api:app --host 0.0.0.0 --port "$PORT"
    ;;
  *)
    exec "$@"
    ;;
esac
