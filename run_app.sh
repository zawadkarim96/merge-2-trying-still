#!/usr/bin/env bash
set -euo pipefail

# Move to the directory containing this script so relative paths resolve correctly.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "Python is required but was not found on PATH." >&2
        exit 1
    fi
fi

exec "$PYTHON_BIN" run_app.py
