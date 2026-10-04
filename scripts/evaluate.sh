#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec conda run --no-capture-output -n vlarobot python -m src.evaluation.evaluate "$@"
