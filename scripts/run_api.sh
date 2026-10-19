#!/usr/bin/env bash
# Run SafeAgent classification API on Virtech or local environment
# Binds to 0.0.0.0 on port 8080 by default (mapped through Virtech NAT).

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

echo "Starting SafeAgent API on ${HOST}:${PORT}..."
exec uv run --frozen uvicorn taed2_safeagent.api.app:app --host "$HOST" --port "$PORT"

