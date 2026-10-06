#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
make
python -m pytest tests -q
python scripts/check_evidence.py
