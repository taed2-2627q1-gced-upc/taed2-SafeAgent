#!/usr/bin/env bash
# Manage the SafeAgent API service (start, stop, restart, status) on Virtech.

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
PID_FILE="${PID_FILE:-${PROJ_DIR}/api.pid}"
LOG_FILE="${LOG_FILE:-${PROJ_DIR}/api.log}"

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_service() {
    if is_running; then
        echo "SafeAgent API is already running (PID: $(cat "$PID_FILE"))."
        return 0
    fi

    echo "Starting SafeAgent API on ${HOST}:${PORT}..."
    nohup uv run --frozen uvicorn taed2_safeagent.api.app:app --host "$HOST" --port "$PORT" >> "$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"

    # Wait briefly and verify process didn't exit immediately
    sleep 2
    if kill -0 "$new_pid" 2>/dev/null; then
        echo "SafeAgent API started successfully (PID: $new_pid). Logs: $LOG_FILE"
    else
        echo "Error: SafeAgent API failed to start. Check $LOG_FILE for details."
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop_service() {
    if ! is_running; then
        echo "SafeAgent API is not running."
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"
    echo "Stopping SafeAgent API (PID: $pid)..."
    kill "$pid" 2>/dev/null || true

    local count=0
    while kill -0 "$pid" 2>/dev/null && [ "$count" -lt 10 ]; do
        sleep 0.5
        count=$((count + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "Process did not stop gracefully, sending SIGKILL..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo "SafeAgent API stopped."
}

status_service() {
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "SafeAgent API is running (PID: $pid) on ${HOST}:${PORT}."
        
        # Test local health endpoint if curl is available
        if command -v curl &>/dev/null; then
            local check_host="127.0.0.1"
            if [ "$HOST" != "0.0.0.0" ]; then
                check_host="$HOST"
            fi
            echo -n "Health check: "
            curl -s "http://${check_host}:${PORT}/health" || echo "failed to connect"
            echo ""
        fi
    else
        echo "SafeAgent API is not running."
        return 1
    fi
}

case "${1:-}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        echo "Restarting SafeAgent API..."
        stop_service
        start_service
        ;;
    status)
        status_service
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

