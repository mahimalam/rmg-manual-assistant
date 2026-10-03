#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/streamlit ]]; then
    echo "Install the environment using the commands in README.md first." >&2
    exit 1
fi
exec .venv/bin/streamlit run main.py
