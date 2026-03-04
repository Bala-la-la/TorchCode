#!/bin/bash
set -e

NOTEBOOKS_DIR=/app/notebooks
TEMPLATES_DIR=/app/templates
SOLUTIONS_DIR=/app/solutions

mkdir -p "$NOTEBOOKS_DIR"

# Blank templates: always overwrite (reset on every start)
echo "📋 Resetting blank notebooks..."
for f in "$TEMPLATES_DIR"/*.ipynb; do
    [ -f "$f" ] || continue
    cp -f "$f" "$NOTEBOOKS_DIR/$(basename "$f")"
done

# Solutions: always overwrite (kept in sync with image)
echo "📖 Syncing solution notebooks..."
for f in "$SOLUTIONS_DIR"/*.ipynb; do
    [ -f "$f" ] || continue
    cp -f "$f" "$NOTEBOOKS_DIR/$(basename "$f")"
done

echo "✅ Notebooks ready — launching JupyterLab"
exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --notebook-dir="$NOTEBOOKS_DIR"
