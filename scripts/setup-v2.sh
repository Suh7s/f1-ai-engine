#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3.13 -m venv .venv-v2
.venv-v2/bin/python -m pip install 'pip==26.2.1'
.venv-v2/bin/python -m pip install -r requirements-v2.lock
.venv-v2/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv-v2/bin/python -m pip check
.venv-v2/bin/f1engine doctor
printf '\nReady. Activate with: . .venv-v2/bin/activate\nThen launch: f1engine app\nOffline preview: f1engine app --demo\n'
