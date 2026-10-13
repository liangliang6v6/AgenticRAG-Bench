#!/usr/bin/env bash
# Download DQA (Decision QA) dataset for PlanRAG.
# Source: https://github.com/myeon9h/PlanRAG (data/ directory)
# Scenarios: LocatE (Europa Universalis IV), ControlE (Victoria 3)
set -euo pipefail

OUT="${1:-$(dirname "$0")/dqa}"
mkdir -p "$OUT"

REPO_URL="https://github.com/myeon9h/PlanRAG"
echo "Cloning PlanRAG repo to get DQA dataset..."
TMP="$(mktemp -d)"
git clone --depth 1 "$REPO_URL" "$TMP/PlanRAG"

# Copy DQA data
if [[ -d "$TMP/PlanRAG/data" ]]; then
    cp -r "$TMP/PlanRAG/data"/* "$OUT/"
    echo "Copied DQA data to $OUT"
else
    echo "ERROR: data/ directory not found in PlanRAG repo"
    ls "$TMP/PlanRAG/"
fi

rm -rf "$TMP"
echo "Done. Files in $OUT"
echo ""
echo "DQA contains two scenarios:"
echo "  LocatE: decision-making in Europa Universalis IV (graph database)"
echo "  ControlE: decision-making in Victoria 3 (relational database)"
echo ""
echo "See methods/planrag/README.md for setup instructions (requires Neo4j / MySQL)."
