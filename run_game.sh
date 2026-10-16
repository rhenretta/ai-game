#!/usr/bin/env bash
# ==============================================================================
# run_game.sh — Single-Command macOS Launcher
# Autonomous Game Engine & Showrunner Harness
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Virtual Environment Auto-Detection and Activation
VENV_PYTHON=""
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ]; then
    VENV_PYTHON="${VIRTUAL_ENV}/bin/python3"
elif [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
    VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -x "$SCRIPT_DIR/venv/bin/python3" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/venv/bin/activate"
fi

if [ -z "$VENV_PYTHON" ]; then
    echo "Virtual environment (.venv) not found. Checking system python3..."
    if command -v python3 >/dev/null 2>&1; then
        echo "Creating virtual environment in $SCRIPT_DIR/.venv..."
        python3 -m venv "$SCRIPT_DIR/.venv"
        # shellcheck source=/dev/null
        source "$SCRIPT_DIR/.venv/bin/activate"
        if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
            echo "Installing dependencies from requirements.txt..."
            pip install --quiet --upgrade pip
            pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
        fi
        VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
    else
        echo "Error: python3 is not installed or not available in PATH." >&2
        exit 1
    fi
fi

# 2. macOS Environment & Cocoa Display Check
OS_NAME="$(uname -s)"
if [ "$OS_NAME" != "Darwin" ]; then
    echo "Notice: Autonomous Game Engine & Showrunner is optimized for macOS (Darwin). Detected: $OS_NAME"
fi

IS_HEADLESS=false
for arg in "$@"; do
    if [ "$arg" = "--headless" ]; then
        IS_HEADLESS=true
        break
    fi
done

if [ "$IS_HEADLESS" = true ]; then
    export SDL_VIDEODRIVER="dummy"
    export SDL_AUDIODRIVER="dummy"
else
    # On macOS, WindowServer provides Cocoa display even without an X11 DISPLAY variable.
    if [ -z "${DISPLAY:-}" ] && [ "$OS_NAME" = "Darwin" ]; then
        export SDL_VIDEODRIVER="cocoa"
    fi
fi

# 3. Signal Trapping (SIGINT / SIGTERM) for Clean SQLite WAL Flush & Terminal Restoration
GAME_PID=""

cleanup() {
    echo ""
    echo "[Launcher] Signal received. Initiating graceful shutdown..."
    if [ -n "$GAME_PID" ] && kill -0 "$GAME_PID" 2>/dev/null; then
        kill -TERM "$GAME_PID" 2>/dev/null || true
        wait "$GAME_PID" 2>/dev/null || true
    fi

    # Explicit SQLite WAL flush if sqlite3 is installed and telemetry database exists
    if command -v sqlite3 >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/data/telemetry.db" ]; then
        sqlite3 "$SCRIPT_DIR/data/telemetry.db" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true
    fi

    # Restore terminal state in case keyboard raw mode or curses was active
    stty sane 2>/dev/null || true
    tput cnorm 2>/dev/null || true
    echo "[Launcher] Shutdown completed cleanly."
    exit 0
}

trap cleanup INT TERM

# 4. Process Execution & Transparent Argument Passing
echo "============================================================"
echo "Autonomous Game Engine & Showrunner Harness"
echo "Native macOS 60 FPS Event Loop | Continuous SQLite WAL"
echo "Python: $VENV_PYTHON"
echo "============================================================"

# If no arguments are passed, enable background showrunner by default
if [ $# -eq 0 ]; then
    "$VENV_PYTHON" "$SCRIPT_DIR/run_game.py" --showrunner &
else
    "$VENV_PYTHON" "$SCRIPT_DIR/run_game.py" "$@" &
fi

GAME_PID=$!
wait "$GAME_PID"
EXIT_CODE=$?
stty sane 2>/dev/null || true
exit $EXIT_CODE
