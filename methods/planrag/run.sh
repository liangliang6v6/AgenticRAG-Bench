#!/usr/bin/env bash
# Run PlanRAG (Plan-then-Retrieval Augmented Generation for Decision Making).
#
# Paper: NAACL 2024 — Lee et al.
# "PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative LLMs
#  as Decision Makers"
# Official repo: https://github.com/myeon9h/PlanRAG
#
# *** IMPORTANT ***
# PlanRAG is NOT a text-retrieval method.
# It uses structured databases (Neo4j or MySQL) for decision-making in
# video game scenarios (EU4 and Victoria 3).
# This differs from all other methods in AgenticRAG-Bench.
#
# Usage:
#   # LocatE scenario (Neo4j / Europa Universalis IV)
#   bash methods/planrag/run.sh \
#       --scenario  locate \
#       --output    results/planrag/ \
#       [--db-uri   bolt://localhost:7687] \
#       [--model    gpt-4o-mini]
#
#   # ControlE scenario (MySQL / Victoria 3)
#   bash methods/planrag/run.sh \
#       --scenario  controle \
#       --output    results/planrag/ \
#       [--mysql-host localhost] \
#       [--model    gpt-4o-mini]
#
# Database setup:
#   git clone https://github.com/myeon9h/PlanRAG
#   # Follow README to set up Neo4j (LocatE) or MySQL (ControlE)
#   # and load the DQA dataset
#
# Environment variables:
#   OPENAI_API_KEY      — OpenAI API key (or use --api-key)
#   PLANRAG_MODEL       — LLM model name (default: gpt-4o-mini)
#   PLANRAG_BASE_URL    — OpenAI-compatible base URL (for local LLM)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- defaults ----------
SCENARIO=""
OUTPUT_DIR=""
MODEL="${PLANRAG_MODEL:-gpt-4o-mini}"
API_KEY="${OPENAI_API_KEY:-}"
BASE_URL="${PLANRAG_BASE_URL:-}"
# Neo4j
DB_URI="bolt://localhost:7687"
DB_USER="neo4j"
DB_PASSWORD="password"
# MySQL
MYSQL_HOST="localhost"
MYSQL_PORT=3306
MYSQL_USER="root"
MYSQL_PASSWORD="password"
MYSQL_DB="victoria3"
# Pipeline
MAX_ROUNDS=5

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)     SCENARIO="$2";     shift 2 ;;
        --output)       OUTPUT_DIR="$2";   shift 2 ;;
        --model)        MODEL="$2";        shift 2 ;;
        --api-key)      API_KEY="$2";      shift 2 ;;
        --base-url)     BASE_URL="$2";     shift 2 ;;
        --db-uri)       DB_URI="$2";       shift 2 ;;
        --db-user)      DB_USER="$2";      shift 2 ;;
        --db-password)  DB_PASSWORD="$2";  shift 2 ;;
        --mysql-host)   MYSQL_HOST="$2";   shift 2 ;;
        --mysql-port)   MYSQL_PORT="$2";   shift 2 ;;
        --mysql-user)   MYSQL_USER="$2";   shift 2 ;;
        --mysql-password) MYSQL_PASSWORD="$2"; shift 2 ;;
        --mysql-db)     MYSQL_DB="$2";     shift 2 ;;
        --max-rounds)   MAX_ROUNDS="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$SCENARIO" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash run.sh --scenario <locate|controle> --output <dir> [options]"
    exit 1
fi

# Set input file based on scenario
case "$SCENARIO" in
    locate)   INPUT_FILE="$BENCH_ROOT/data/dqa/locate.jsonl" ;;
    controle) INPUT_FILE="$BENCH_ROOT/data/dqa/controle.jsonl" ;;
    *) echo "ERROR: Unknown scenario '$SCENARIO'. Choose: locate or controle"; exit 1 ;;
esac

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Download with: bash data/download.sh dqa"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/$SCENARIO"
PREDICTIONS="$OUTPUT_DIR/$SCENARIO/predictions.jsonl"

# Build extra args
EXTRA_ARGS=()
[[ -n "$API_KEY" ]]   && EXTRA_ARGS+=(--api-key "$API_KEY")
[[ -n "$BASE_URL" ]]  && EXTRA_ARGS+=(--base-url "$BASE_URL")

echo "=== PlanRAG: $SCENARIO ==="
echo "  Model:       $MODEL"
echo "  Input:       $INPUT_FILE"
echo "  Output:      $PREDICTIONS"
echo ""

if [[ "$SCENARIO" == "locate" ]]; then
    python "$SCRIPT_DIR/src/infer.py" \
        --input         "$INPUT_FILE" \
        --output        "$PREDICTIONS" \
        --scenario      locate \
        --model         "$MODEL" \
        --db-uri        "$DB_URI" \
        --db-user       "$DB_USER" \
        --db-password   "$DB_PASSWORD" \
        --max-rounds    "$MAX_ROUNDS" \
        "${EXTRA_ARGS[@]}"
else
    python "$SCRIPT_DIR/src/infer.py" \
        --input          "$INPUT_FILE" \
        --output         "$PREDICTIONS" \
        --scenario       controle \
        --model          "$MODEL" \
        --mysql-host     "$MYSQL_HOST" \
        --mysql-port     "$MYSQL_PORT" \
        --mysql-user     "$MYSQL_USER" \
        --mysql-password "$MYSQL_PASSWORD" \
        --mysql-db       "$MYSQL_DB" \
        --max-rounds     "$MAX_ROUNDS" \
        "${EXTRA_ARGS[@]}"
fi

echo ""
echo "=== Evaluating ==="
python "$BENCH_ROOT/eval/eval.py" \
    --predictions "$PREDICTIONS" \
    --output      "$OUTPUT_DIR/$SCENARIO/metrics.json"
