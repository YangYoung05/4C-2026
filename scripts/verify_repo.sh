#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m py_compile "$ROOT"/雷霆医疗队/01_main/*.py
python3 "$ROOT/雷霆医疗队/01_main/a_foundation.py" \
  --project-root "$ROOT/雷霆医疗队" \
  --check

cd "$ROOT/Global Map"
npm ci
npm run verify

printf 'Repository verification passed.\n'
