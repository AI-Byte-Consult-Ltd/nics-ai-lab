"""Turn a backup folder made by db_backup.py into one SQL file for psql.

  python scripts/db_restore.py <backup folder> --out restore.sql
  psql "<connection string of an EMPTY database>" -v ON_ERROR_STOP=1 -1 -f restore.sql

Order inside the file: schema (tables, functions, keys, indexes, views) -> data ->
foreign keys and triggers -> sequence positions. Data goes in as
INSERT ... SELECT ... FROM json_populate_recordset(...), so every Postgres type round-trips
(numeric, timestamptz, jsonb, arrays, bytea). Generated columns are skipped, identity
columns keep their values (OVERRIDING SYSTEM VALUE).

app_secrets comes back with "__REDACTED__" in place of secret values: put the real
values in again (or rotate them) before the restored system is used.
Never run this against the live production database. Standard library only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import secrets
import sys
from pathlib import Path


def qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def dollar(text: str) -> str:
    while True:
        tag = f"$n{secrets.token_hex(4)}$"
        if tag not in text:
            return f"{tag}{text}{tag}"


def table_inserts(name: str, path: Path, flags: dict, batch: int):
    skip = set(flags.get("generated") or [])
    overriding = " OVERRIDING SYSTEM VALUE" if flags.get("identity_always") else ""
    rows, cols = [], None

    def flush():
        if not rows:
            return None
        payload = "[" + ",".join(rows) + "]"
        col_sql = ", ".join(qi(c) for c in cols)
        return (f"INSERT INTO public.{qi(name)} ({col_sql}){overriding}\n"
                f"SELECT {col_sql} FROM json_populate_recordset(NULL::public.{qi(name)}, {dollar(payload)}::json);\n")

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if cols is None:
                cols = [c for c in json.loads(line).keys() if c not in skip]
            rows.append(line)
            if len(rows) >= batch:
                yield flush()
                rows = []
    sql = flush()
    if sql:
        yield sql


def build(folder: Path, out, batch: int = 2000) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("errors"):
        sys.exit(f"backup has failed tables: {sorted(manifest['errors'])}")
    flags = {f["table"]: f for f in json.loads((folder / "schema" / "column_flags.json").read_text(encoding="utf-8"))}
    out.write("-- NICS Trading DB restore, generated from " + folder.name + "\n")
    out.write("SET client_min_messages = warning;\n")
    out.write((folder / "schema" / "pre_data.sql").read_text(encoding="utf-8") + "\n\n")
    counts = {}
    for name in sorted(manifest["tables"]):
        out.write(f"\n-- data: {name}\n")
        for stmt in table_inserts(name, folder / "data" / f"{name}.jsonl.gz", flags.get(name, {}), batch):
            out.write(stmt)
        counts[name] = manifest["tables"][name]["rows"]
    out.write("\n")
    out.write((folder / "schema" / "post_data.sql").read_text(encoding="utf-8") + "\n\n")
    out.write((folder / "schema" / "sequences.sql").read_text(encoding="utf-8") + "\n")
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default="-", help="SQL file (.sql or .sql.gz); '-' = stdout")
    ap.add_argument("--batch", type=int, default=2000, help="rows per INSERT (default 2000)")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    if args.out == "-":
        build(folder, sys.stdout, args.batch)
    elif args.out.endswith(".gz"):
        with gzip.open(args.out, "wt", encoding="utf-8") as f:
            counts = build(folder, f, args.batch)
        print(f"wrote {args.out}: {len(counts)} tables, {sum(counts.values())} rows", file=sys.stderr)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            counts = build(folder, f, args.batch)
        print(f"wrote {args.out}: {len(counts)} tables, {sum(counts.values())} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
