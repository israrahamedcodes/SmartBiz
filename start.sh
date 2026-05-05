#!/usr/bin/env bash
set -e

echo "==> Python: $(python --version)"
echo "==> Pip: $(pip --version)"
echo "==> Working dir: $(pwd)"
echo "==> Files: $(ls)"

# Always install from backend requirements as a safety net
pip install -r backend/requirements.txt --quiet

echo "==> Gunicorn check: $(python -m gunicorn --version)"

cd backend
echo "==> Running from: $(pwd)"
exec python -m gunicorn app:app --bind "0.0.0.0:${PORT:-5000}" --workers 1 --timeout 120
