#!/usr/bin/env bash
# Smoke test: run PlanRAG on 1 example per scenario.
# Requires: Neo4j (locate) or MySQL (controle) running with DQA data loaded.
# Usage: bash methods/planrag/test.sh [--scenario locate|controle]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${PLANRAG_MODEL:-gpt-4o-mini}"
API_KEY="${OPENAI_API_KEY:-}"
BASE_URL="${PLANRAG_BASE_URL:-}"
DB_URI="${PLANRAG_NEO4J_URI:-bolt://localhost:7687}"
DB_USER="${PLANRAG_NEO4J_USER:-neo4j}"
DB_PASSWORD="${PLANRAG_NEO4J_PASSWORD:-password}"
MYSQL_HOST="${PLANRAG_MYSQL_HOST:-localhost}"
MYSQL_USER="${PLANRAG_MYSQL_USER:-root}"
MYSQL_PASSWORD="${PLANRAG_MYSQL_PASSWORD:-password}"
MYSQL_DB="${PLANRAG_MYSQL_DB:-victoria3}"
TEST_SCENARIO="${1:-}"  # optional: locate or controle

EXTRA_ARGS=()
[[ -n "$API_KEY" ]]  && EXTRA_ARGS+=(--api-key "$API_KEY")
[[ -n "$BASE_URL" ]] && EXTRA_ARGS+=(--base-url "$BASE_URL")

run_scenario() {
    local SCENARIO="$1"
    local FULL_INPUT="$BENCH_ROOT/data/dqa/${SCENARIO}.jsonl"
    [[ -f "$FULL_INPUT" ]] || { echo "SKIP $SCENARIO — data not found at $FULL_INPUT"; return; }

    SMOKE_INPUT="/tmp/planrag_smoke_${SCENARIO}.jsonl"
    head -1 "$FULL_INPUT" > "$SMOKE_INPUT"
    SMOKE_OUT="/tmp/planrag_smoke_out_${SCENARIO}"
    mkdir -p "$SMOKE_OUT"

    echo "=== Smoke test: PlanRAG on $SCENARIO (1 example) ==="
    if [[ "$SCENARIO" == "locate" ]]; then
        python "$SCRIPT_DIR/src/infer.py" \
            --input       "$SMOKE_INPUT" \
            --output      "$SMOKE_OUT/predictions.jsonl" \
            --scenario    locate \
            --model       "$MODEL" \
            --db-uri      "$DB_URI" \
            --db-user     "$DB_USER" \
            --db-password "$DB_PASSWORD" \
            "${EXTRA_ARGS[@]}" && echo "PASS: locate" || echo "FAIL: locate"
    else
        python "$SCRIPT_DIR/src/infer.py" \
            --input           "$SMOKE_INPUT" \
            --output          "$SMOKE_OUT/predictions.jsonl" \
            --scenario        controle \
            --model           "$MODEL" \
            --mysql-host      "$MYSQL_HOST" \
            --mysql-user      "$MYSQL_USER" \
            --mysql-password  "$MYSQL_PASSWORD" \
            --mysql-db        "$MYSQL_DB" \
            "${EXTRA_ARGS[@]}" && echo "PASS: controle" || echo "FAIL: controle"
    fi
}

if [[ -n "$TEST_SCENARIO" ]]; then
    run_scenario "$TEST_SCENARIO"
else
    run_scenario "locate"
    run_scenario "controle"
fi

echo "Smoke test done."
