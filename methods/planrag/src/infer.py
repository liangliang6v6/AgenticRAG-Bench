#!/usr/bin/env python3
"""PlanRAG: Plan-then-Retrieval Augmented Generation for Decision Making.

Official source:
  https://github.com/myeon9h/PlanRAG
  Paper: NAACL 2024 — Lee et al.
  "PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative LLMs
   as Decision Makers"

Algorithm (Plan-Thought-Action-Observation loop):
  1. Given: decision question, business rules, and a structured database
     (Neo4j graph DB for LocatE, MySQL relational DB for ControlE).
  2. Planning step: LLM generates a plan — what data analyses are needed.
  3. Retrieval step: LLM generates database queries (Cypher or SQL).
  4. Observation: Execute queries and return results.
  5. Assess if more analysis is needed; repeat or make the final decision.

This differs from all other methods in AgenticRAG-Bench:
  - No text corpus retrieval — uses structured databases
  - Decision-making task (game scenario choices), not free-text QA
  - Requires running Neo4j (LocatE) or MySQL (ControlE) with loaded data

Prompt source:
  'Plan'-'Thought'-'Action'-'Observation' structure from myeon9h/PlanRAG.
  Source: arxiv.org/abs/2406.12430 (NAACL 2024), Figure 2 and Section 3.

Usage:
    # LocatE scenario (Neo4j)
    python src/infer.py \\
        --input   ../../data/dqa/locate.jsonl \\
        --output  /tmp/planrag_locate.jsonl \\
        --scenario locate \\
        --db-uri  bolt://localhost:7687 \\
        --db-user neo4j --db-password password \\
        --model   gpt-4o-mini \\
        --api-key sk-...

    # ControlE scenario (MySQL)
    python src/infer.py \\
        --input    ../../data/dqa/controle.jsonl \\
        --output   /tmp/planrag_controle.jsonl \\
        --scenario controle \\
        --mysql-host localhost --mysql-port 3306 \\
        --mysql-user root --mysql-password password \\
        --mysql-db  victoria3 \\
        --model    gpt-4o-mini \\
        --api-key  sk-...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Prompt templates
# Source: myeon9h/PlanRAG — Plan-Thought-Action-Observation format
# Reference: arxiv.org/abs/2406.12430, Figure 2, Section 3
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a decision maker. Given a decision question, business rules, and access to a database, \
your job is to make the best decision.

You will follow this structured reasoning process:
Plan: What data analyses are needed to make this decision?
Thought: What specific query should I run to get this data?
Action: [Execute a database query]
Observation: [Results from the database]
... (repeat Plan/Thought/Action/Observation as needed)
Decision: The best answer is [ANSWER] because [REASON].
"""

PLAN_PROMPT_TMPL = """\
Decision Question: {question}

Business Rules:
{rules}

Database Schema:
{schema}

Based on the question and business rules, make a plan for what data you need to analyze \
to reach a decision. Then generate a query to retrieve the first piece of data.

Plan: {plan_prefix}"""

FOLLOW_UP_PROMPT_TMPL = """\
Previous observations:
{observations}

Based on your previous plan and observations, decide:
1. If you need more data, generate another query.
2. If you have enough information, make your final decision.

Thought:"""

DECISION_PROMPT_TMPL = """\
Decision Question: {question}

Business Rules:
{rules}

Data gathered:
{observations}

Based on the data and business rules, make your final decision.
Your answer must be one of the valid options for this question.
Format: Decision: The best answer is [ANSWER]."""


# ---------------------------------------------------------------------------
# Database execution
# ---------------------------------------------------------------------------

def execute_neo4j_query(query: str, driver) -> str:
    """Execute a Cypher query against Neo4j and return formatted results."""
    try:
        with driver.session() as session:
            result = session.run(query)
            records = [dict(r) for r in result]
            if not records:
                return "No results found."
            return json.dumps(records[:20], indent=2, default=str)
    except Exception as exc:
        return f"Query error: {exc}"


def execute_mysql_query(query: str, connection) -> str:
    """Execute a SQL query against MySQL and return formatted results."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchmany(20)
            cols = [d[0] for d in cursor.description] if cursor.description else []
            if not rows:
                return "No results found."
            records = [dict(zip(cols, row)) for row in rows]
            return json.dumps(records, indent=2, default=str)
    except Exception as exc:
        return f"Query error: {exc}"


def extract_query(text: str, scenario: str) -> str | None:
    """Extract a database query from model output."""
    if scenario == "locate":
        # Cypher query — extract from Action: block or backtick code block
        m = re.search(r"(?:Action:|```cypher?)\s*(MATCH[\s\S]*?)(?:```|$)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: any MATCH ... RETURN pattern
        m = re.search(r"(MATCH\s+[\s\S]*?RETURN\s+[^\n]+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    else:
        # SQL query
        m = re.search(r"(?:Action:|```sql?)\s*(SELECT[\s\S]*?)(?:```|;|$)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"(SELECT\s+[\s\S]*?FROM\s+[^\n]+(?:\s+WHERE\s+[^\n]+)?)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def extract_final_decision(text: str) -> str:
    """Extract the final decision from model output."""
    m = re.search(r"(?:Decision:|best answer is)\s*(.+?)(?:\n|$|because)", text, re.IGNORECASE)
    if m:
        answer = m.group(1).strip().strip("[].")
        return answer
    # Last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict], model: str, api_key: str, base_url: str | None, max_tokens: int = 512) -> str:
    """Call an OpenAI-compatible LLM API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# PlanRAG algorithm
# ---------------------------------------------------------------------------

def get_db_schema(scenario: str, driver_or_conn) -> str:
    """Get database schema description."""
    if scenario == "locate":
        try:
            with driver_or_conn.session() as session:
                result = session.run("CALL apoc.meta.schema()")
                schema = [dict(r) for r in result]
                return json.dumps(schema[:5], indent=2, default=str)
        except Exception:
            return "(Neo4j graph database — use MATCH queries to explore)"
    else:
        try:
            schema_lines = []
            with driver_or_conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = [row[0] for row in cursor.fetchall()]
                for table in tables[:10]:
                    cursor.execute(f"DESCRIBE {table}")
                    cols = [row[0] for row in cursor.fetchall()]
                    schema_lines.append(f"Table {table}: {', '.join(cols)}")
            return "\n".join(schema_lines)
        except Exception:
            return "(MySQL relational database)"


def run_planrag(
    question: str,
    rules: str,
    scenario: str,
    db_conn,
    model: str,
    api_key: str,
    base_url: str | None,
    max_rounds: int = 5,
) -> dict[str, Any]:
    """Run the PlanRAG decision-making loop.

    Source: myeon9h/PlanRAG — Plan-Thought-Action-Observation loop
    Reference: arxiv.org/abs/2406.12430, Section 3.2
    """
    schema = get_db_schema(scenario, db_conn)
    observations: list[str] = []
    reasoning_trace: list[dict] = []

    # Initial planning
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PLAN_PROMPT_TMPL.format(
            question=question, rules=rules, schema=schema, plan_prefix=""
        )},
    ]
    response = call_llm(messages, model, api_key, base_url)
    reasoning_trace.append({"step": "plan", "content": response})

    for round_num in range(max_rounds):
        query = extract_query(response, scenario)
        if not query:
            break

        # Execute query
        if scenario == "locate":
            obs = execute_neo4j_query(query, db_conn)
        else:
            obs = execute_mysql_query(query, db_conn)

        observations.append(f"Query {round_num+1}: {query}\nResult: {obs}")
        reasoning_trace.append({"step": "action", "query": query, "observation": obs})

        # Check if we have enough to decide
        obs_str = "\n\n".join(observations)
        follow_up_prompt = FOLLOW_UP_PROMPT_TMPL.format(observations=obs_str)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": follow_up_prompt})
        response = call_llm(messages, model, api_key, base_url)
        reasoning_trace.append({"step": "thought", "content": response})

        if "decision:" in response.lower() or "best answer" in response.lower():
            break

    # Final decision
    if not any("decision:" in t["content"].lower() for t in reasoning_trace if "content" in t):
        obs_str = "\n\n".join(observations) if observations else "No data retrieved."
        decision_prompt = DECISION_PROMPT_TMPL.format(
            question=question, rules=rules, observations=obs_str
        )
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": decision_prompt})
        response = call_llm(messages, model, api_key, base_url, max_tokens=256)
        reasoning_trace.append({"step": "final_decision", "content": response})

    final_answer = extract_final_decision(
        next((t["content"] for t in reversed(reasoning_trace) if "content" in t), "")
    )
    return {"answer": final_answer, "trace": reasoning_trace}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True, type=Path)
    parser.add_argument("--output",      required=True, type=Path)
    parser.add_argument("--scenario",    required=True, choices=["locate", "controle"],
                        help="locate (Neo4j/EU4) or controle (MySQL/Victoria3)")
    # LLM
    parser.add_argument("--model",       default="gpt-4o-mini")
    parser.add_argument("--api-key",     default=None, help="OpenAI API key (or env OPENAI_API_KEY)")
    parser.add_argument("--base-url",    default=None, help="OpenAI-compatible base URL (for local LLM)")
    # Neo4j (locate)
    parser.add_argument("--db-uri",      default="bolt://localhost:7687")
    parser.add_argument("--db-user",     default="neo4j")
    parser.add_argument("--db-password", default="password")
    # MySQL (controle)
    parser.add_argument("--mysql-host",  default="localhost")
    parser.add_argument("--mysql-port",  type=int, default=3306)
    parser.add_argument("--mysql-user",  default="root")
    parser.add_argument("--mysql-password", default="password")
    parser.add_argument("--mysql-db",    default="victoria3")
    # Pipeline
    parser.add_argument("--max-rounds",  type=int, default=5)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    # Connect to database
    if args.scenario == "locate":
        try:
            from neo4j import GraphDatabase
        except ImportError:
            print("ERROR: pip install neo4j")
            sys.exit(1)
        db_conn = GraphDatabase.driver(args.db_uri, auth=(args.db_user, args.db_password))
        print(f"Connected to Neo4j at {args.db_uri}")
    else:
        try:
            import pymysql
        except ImportError:
            print("ERROR: pip install pymysql")
            sys.exit(1)
        db_conn = pymysql.connect(
            host=args.mysql_host, port=args.mysql_port,
            user=args.mysql_user, password=args.mysql_password,
            database=args.mysql_db, cursorclass=pymysql.cursors.DictCursor,
        )
        print(f"Connected to MySQL at {args.mysql_host}:{args.mysql_port}/{args.mysql_db}")

    records = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    done_ids: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                done_ids.add(str(json.loads(line)["id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            if str(rec.get("id", "")) in done_ids:
                continue
            print(f"[{i+1}/{len(records)}] {rec.get('question', '')[:80]}", flush=True)
            result = run_planrag(
                question=rec.get("question", ""),
                rules=rec.get("rules", rec.get("business_rules", "")),
                scenario=args.scenario,
                db_conn=db_conn,
                model=args.model,
                api_key=api_key,
                base_url=args.base_url,
                max_rounds=args.max_rounds,
            )
            gold = rec.get("answer", rec.get("answers", ""))
            gold_str = gold[0] if isinstance(gold, list) and gold else (gold or "")
            out_rec = {
                "id": str(rec.get("id", i)),
                "question": rec.get("question", ""),
                "prediction": result["answer"],
                "answer": gold_str,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

    if args.scenario == "locate":
        db_conn.close()
    else:
        db_conn.close()

    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
