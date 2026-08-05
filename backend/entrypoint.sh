#!/bin/sh
# entrypoint.sh — run DB migrations then start the server
set -e

echo "Running Alembic migrations..."
alembic -c /app/alembic.ini upgrade head

echo "Starting uvicorn..."
exec python -m uvicorn app.Main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers "${WORKERS:-2}"
