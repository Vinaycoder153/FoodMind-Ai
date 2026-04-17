#!/usr/bin/env bash
# Start the FoodMind AI backend
set -e
cd "$(dirname "$0")/backend"

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt --quiet
fi

echo "Starting FoodMind AI backend on http://localhost:8000"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
