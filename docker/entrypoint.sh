#!/bin/sh
set -e
python manage.py init-db
if [ "${ERAG_SEED_DEMO:-0}" = "1" ]; then python manage.py seed-demo; fi
exec uvicorn api:app --host 0.0.0.0 --port 8000
