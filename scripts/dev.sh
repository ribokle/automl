#!/usr/bin/env bash
# Launch FastAPI + Next.js dev servers together.
set -euo pipefail

# Start FastAPI in the background.
uv run uvicorn api.main:app --reload --port 8000 &
API_PID=$!

# Start Next.js.
( cd web && pnpm dev ) &
WEB_PID=$!

trap 'kill $API_PID $WEB_PID 2>/dev/null || true' EXIT
wait
