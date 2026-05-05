#!/usr/bin/env bash
set -e

echo "==> Python version: $(python --version)"
echo "==> Working dir: $(pwd)"
echo "==> Gunicorn path: $(which gunicorn || echo NOT FOUND)"

# Install deps if gunicorn is missing (safety net)
if ! command -v gunicorn &> /dev/null; then
    echo "==> gunicorn not found, installing..."
    pip install gunicorn
fi

cd backend
echo "==> Starting gunicorn from: $(pwd)"
exec gunicorn app:app --bind "0.0.0.0:${PORT:-5000}" --workers 1 --timeout 120
