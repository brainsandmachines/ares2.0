"""Read-only table viewer for the orchestrator SQLite database.

Examples:
    python orchestrator/show_db.py
    python orchestrator/show_db.py --active          # models a controller is running now
    python orchestrator/show_db.py --status TRAINING
    python orchestrator/show_db.py --format markdown
    python orchestrator/show_db.py --all-columns --limit 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path


DEFAULT_COLUMNS = (
    "model_id",
    "status",
    "current_stage",
    "cluster_node",
    "slurm_job_id",
    "arch",
    "protocol",
    "eps",
    "epoch",
    "next_epoch",
    "target_epoch",
    "requeued",
    "best_checkpoint",
    "best_score",
)


def default_db_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / "orchestrator" / ".env"
    dotenv = read_dotenv(env_path)
    return Path(
        os.environ.get(
            "ORCH_DB",
            dotenv.get("ORCH_DB", str(repo_root / "orchestrator" / "orchestrator.db")),
        )
    )


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"database not found: {db_path}")
    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("PRAGMA table_info(models_queue)").fetchall()
    return [str(r["name"]) for r in rows]


# Statuses a controller can be actively working (mirror of db.ACTIVE_STATUSES).
ACTIVE_STATUSES = ("TRAINING", "AA_EVAL", "PLOTTING")


def fetch_rows(
    conn: sqlite3.Connection,
    *,
    status: str | None,
    limit: int | None,
    all_columns: bool,
    active: bool = False,
) -> tuple[list[str], list[dict[str, object]]]:
    available = table_columns(conn)
    available_set = set(available)
    if all_columns:
        columns = available
        select_sql = "SELECT * FROM models_queue"
    else:
        columns = list(DEFAULT_COLUMNS)
        next_epoch_sql = (
            "next_epoch" if "next_epoch" in available_set else "current_epoch AS next_epoch"
        )
        target_epoch_expr = (
            "COALESCE(final_epoch, warmup_epochs + linear_epochs + plateau_epochs)"
            if "final_epoch" in available_set
            else "warmup_epochs + linear_epochs + plateau_epochs"
        )
        requeued_sql = "requeued" if "requeued" in available_set else "0 AS requeued"
        best_checkpoint_sql = (
            "best_checkpoint"
            if "best_checkpoint" in available_set
            else "NULL AS best_checkpoint"
        )
        best_score_sql = "best_score" if "best_score" in available_set else "NULL AS best_score"
        select_sql = f"""
            SELECT
                model_id,
                status,
                current_stage,
                cluster_node,
                slurm_job_id,
                arch,
                protocol,
                eps,
                current_epoch || '/' || ({target_epoch_expr}) AS epoch,
                {next_epoch_sql},
                {target_epoch_expr} AS target_epoch,
                {requeued_sql},
                {best_checkpoint_sql},
                {best_score_sql}
            FROM models_queue
        """

    params: list[object] = []
    clauses: list[str] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if active:
        # Owned by a live controller task and in a non-terminal stage = running now.
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        clauses.append(f"slurm_job_id IS NOT NULL AND status IN ({placeholders})")
        params.extend(ACTIVE_STATUSES)
    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    order_sql = " ORDER BY priority DESC, model_id ASC"
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT ?"
        params.append(limit)

    rows = conn.execute(select_sql + where_sql + order_sql + limit_sql, params).fetchall()
    return columns, [{c: r[c] for c in columns} for r in rows]


def display(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def print_table(columns: list[str], rows: list[dict[str, object]], max_width: int) -> None:
    rendered = [{c: display(row[c]) for c in columns} for row in rows]
    widths = {
        c: min(max_width, max(len(c), *(len(row[c]) for row in rendered)) if rendered else len(c))
        for c in columns
    }
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for row in rendered:
        print("  ".join(truncate(row[c], widths[c]).ljust(widths[c]) for c in columns))
    print(f"\n{len(rows)} rows")


def print_markdown(columns: list[str], rows: list[dict[str, object]]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [display(row[c]).replace("|", "\\|") for c in columns]
        print("| " + " | ".join(values) + " |")


def print_csv(columns: list[str], rows: list[dict[str, object]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({c: display(row[c]) for c in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Show orchestrator DB rows as a table.")
    parser.add_argument("--db", type=Path, default=default_db_path(), help="SQLite DB path")
    parser.add_argument("--status", help="optional status filter, e.g. TRAINING")
    parser.add_argument(
        "--active",
        action="store_true",
        help="only models a controller is running right now (owned + non-terminal)",
    )
    parser.add_argument("--limit", type=int, help="maximum rows to print")
    parser.add_argument("--all-columns", action="store_true", help="print every DB column")
    parser.add_argument("--max-width", type=int, default=32, help="max width per table cell")
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "csv"),
        default="table",
        help="output format",
    )
    args = parser.parse_args()

    with connect_readonly(args.db) as conn:
        columns, rows = fetch_rows(
            conn,
            status=args.status,
            limit=args.limit,
            all_columns=args.all_columns,
            active=args.active,
        )

    if args.format == "markdown":
        print_markdown(columns, rows)
    elif args.format == "csv":
        print_csv(columns, rows)
    else:
        print_table(columns, rows, args.max_width)


if __name__ == "__main__":
    main()
