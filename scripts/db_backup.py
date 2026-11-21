"""Full backup of the NICS Trading database to this computer (read-only on the server).

Downloads every table of the production database (public schema) through the n8n
workflow "[00 OPS] Database Backup Export" and stores it as one dated folder:

  <out>/2026-11-21_0415/
      manifest.json            what was saved: tables, rows, sizes, timings, errors
      schema/pre_data.sql      tables, functions, sequences, primary keys, indexes, views
      schema/post_data.sql     foreign keys and triggers (applied after the data)
      schema/sequences.sql     sequence positions
      schema/column_flags.json generated / identity columns (used by db_restore.py)
      data/<table>.jsonl.gz    one JSON object per row, in primary-key order

The secret values of app_secrets (keys, tokens, passwords) are NOT in the backup:
the server replaces them with "__REDACTED__". Everything else is a full copy.
n8n Credentials are never exported either.

The folder is written as <name>.partial and renamed only when every table finished,
so a half-done run is never mistaken for a good backup. Old backups beyond --keep
are deleted (only complete ones are counted).

Restore into an empty database:  python scripts/db_restore.py <backup folder>  (see there).

Setup (once, Windows laptop):
  n8n: open "[00 OPS] Database Backup Export", run "Create or Show Backup Key", copy the key.
  Run the script once and paste the key when asked (hidden; saved like setx for the nightly
  task). Never commit it, never paste it into chats. Or: setx NICS_BACKUP_KEY <key>.
  The server address defaults to the NICS n8n (derived from NICS_DATA_URL when set);
  NICS_BACKUP_URL overrides it.

Run:  venv\\Scripts\\python.exe scripts\\db_backup.py
Nightly: scripts\\windows\\install_tasks.bat registers "NICS DB backup" (04:15).
Standard library only.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT.parent / "backups" / "db"
WEBHOOK_PATH = "nics-db-backup"
DEFAULT_URL = f"https://n8n.aibyteconsult.com/webhook/{WEBHOOK_PATH}"


def windows_user_env(name: str) -> str | None:
    """Value saved with `setx` (HKCU\\Environment); setx does not reach windows already open."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, name)[0]).strip() or None
    except (ImportError, OSError):
        return None


def save_windows_user_env(name: str, value: str) -> bool:
    """Same effect as `setx name value`, without the value appearing in the shell history."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
        return True
    except (ImportError, OSError):
        return False


def backup_key() -> str | None:
    """Key from this window, else from setx, else asked once (hidden input) and saved for the nightly task."""
    key = (os.environ.get("NICS_BACKUP_KEY") or "").strip() or windows_user_env("NICS_BACKUP_KEY")
    if key or not sys.stdin.isatty():
        return key
    import getpass
    key = getpass.getpass("Paste the backup key from n8n (hidden, press Enter): ").strip()
    if key and save_windows_user_env("NICS_BACKUP_KEY", key):
        print("key saved for the nightly task (same as setx)")
    return key or None


def backup_url(env: dict | None = None) -> str | None:
    env = os.environ if env is None else env
    if env.get("NICS_BACKUP_URL"):
        return env["NICS_BACKUP_URL"]
    data = env.get("NICS_DATA_URL")
    if not data:
        return DEFAULT_URL
    base = data.rstrip("/").rsplit("/", 1)[0]
    return f"{base}/{WEBHOOK_PATH}"


class Client:
    def __init__(self, url: str, key: str, retries: int = 4, timeout: int = 180):
        self.url, self.key, self.retries, self.timeout = url, key, retries, timeout

    def get(self, params: dict) -> tuple[bytes, dict]:
        """Returns (decompressed body, headers). Retries network errors and 5xx."""
        q = urllib.parse.urlencode(params)
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(f"{self.url}?{q}", headers={"X-NICS-Backup-Key": self.key})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                    headers = {k.lower(): v for k, v in r.headers.items()}
                body = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
                return body, headers
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    sys.exit("403: wrong NICS_BACKUP_KEY (n8n: [00 OPS] Database Backup Export -> "
                             "Create or Show Backup Key).")
                if e.code < 500:
                    raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
                last = e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"request failed after {self.retries} tries: {last}")

    def json(self, params: dict) -> dict:
        body, _ = self.get(params)
        return json.loads(body.decode("utf-8"))


def report(ok: bool, url: str, final: Path | None, error: str = "", log=print) -> bool:
    """Tell the CEO desk how the run went (block «Бэкап»): time, rows, size, folder. No data, no keys.

    Reuses the lab metrics webhook and NICS_DATA_KEY that the laptop already has. Never raises,
    never fails the backup."""
    summary = {"ok": ok, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "error": error[:500]}
    if final is not None:
        try:
            m = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            summary.update(folder=str(final), stamp=final.name, tables=len(m.get("tables") or {}),
                           rows=m.get("total_rows"), bytes=m.get("total_file_bytes"),
                           started_at=m.get("started_at"))
        except (OSError, ValueError):
            pass
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from nics.telemetry.push import push
    except ImportError:
        return False
    env = dict(os.environ)
    env.setdefault("NICS_DATA_KEY", windows_user_env("NICS_DATA_KEY") or "")
    if not env.get("NICS_METRICS_URL") and not env.get("NICS_DATA_URL"):
        p = urllib.parse.urlsplit(url)
        env["NICS_METRICS_URL"] = urllib.parse.urlunsplit((p.scheme, p.netloc, "/webhook/nics-lab-metrics", "", ""))
    text = (f"OK {summary.get('stamp')} rows={summary.get('rows')}" if ok else f"FAILED: {error[:300]}")
    return push("nics-db", "db_backup", summary, text=text, cycle=summary.get("stamp") or "", log=log, env=env)


def chunk_limit(table: dict, target_bytes: int) -> int:
    """Rows per request so that one response stays near target_bytes of raw data."""
    est, size = int(table.get("est_rows") or -1), int(table.get("bytes") or 0)
    if est <= 0:
        return 2000
    per_row = max(size / est, 50)
    return int(min(max(target_bytes / per_row, 100), 50000))


def dump_table(client: Client, table: dict, path: Path, target_bytes: int, log) -> dict:
    name = table["table"]
    has_pk = bool(table.get("pk"))
    limit = chunk_limit(table, target_bytes)
    rows = chunks = 0
    after, offset = "", 0
    t0 = time.time()
    with gzip.open(path, "wb", compresslevel=6) as out:
        while True:
            params = {"mode": "chunk", "table": name, "limit": limit}
            if has_pk and after:
                params["after"] = after
            if not has_pk and offset:
                params["offset"] = offset
            body, headers = client.get(params)
            n = int(headers.get("x-rows") or 0)
            if n == 0:
                break
            text = body.decode("utf-8")
            got = text.count("\n") + 1 if text else 0
            if got != n:
                raise RuntimeError(f"{name}: server said {n} rows, file has {got}")
            out.write(body)
            out.write(b"\n")
            rows += n
            chunks += 1
            if has_pk:
                after = headers.get("x-last-key") or ""
                if not after:
                    raise RuntimeError(f"{name}: no last key for paging")
            else:
                offset += n
            if chunks % 10 == 0:
                log(f"  {name}: {rows} rows so far")
    return {"rows": rows, "chunks": chunks, "limit": limit, "seconds": round(time.time() - t0, 1),
            "file_bytes": path.stat().st_size, "est_rows": table.get("est_rows"), "db_bytes": table.get("bytes"),
            "primary_key": [p["col"] for p in table.get("pk") or []]}


def prune(out_root: Path, keep: int, log) -> None:
    done = sorted(p for p in out_root.iterdir() if p.is_dir() and not p.name.endswith(".partial")
                  and (p / "manifest.json").exists())
    for old in done[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
        log(f"deleted old backup {old.name}")
    for stale in out_root.glob("*.partial"):
        if time.time() - stale.stat().st_mtime > 2 * 86400:
            shutil.rmtree(stale, ignore_errors=True)


def run(client: Client, out_root: Path, keep: int, target_bytes: int, only: list[str] | None, log) -> Path:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    work = out_root / f"{stamp}.partial"
    final = out_root / stamp
    (work / "data").mkdir(parents=True, exist_ok=True)
    (work / "schema").mkdir(exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)

    listing = client.json({"mode": "tables"})
    tables = listing.get("tables") or []
    if not tables:
        raise RuntimeError("server returned no tables")
    if only:
        tables = [t for t in tables if t["table"] in only]
    log(f"{len(tables)} tables")

    schema = client.json({"mode": "schema"})
    (work / "schema" / "pre_data.sql").write_text(schema.get("pre_data") or "", encoding="utf-8")
    (work / "schema" / "post_data.sql").write_text(schema.get("post_data") or "", encoding="utf-8")
    (work / "schema" / "column_flags.json").write_text(
        json.dumps(schema.get("column_flags") or [], indent=1), encoding="utf-8")

    result, errors = {}, {}
    for t in sorted(tables, key=lambda x: int(x.get("bytes") or 0)):
        name = t["table"]
        try:
            result[name] = dump_table(client, t, work / "data" / f"{name}.jsonl.gz", target_bytes, log)
            log(f"{name}: {result[name]['rows']} rows, {result[name]['file_bytes'] // 1024} KB")
        except Exception as e:  # keep going, report at the end
            errors[name] = str(e)[:500]
            log(f"{name}: ERROR {e}")

    # sequence positions are read after the data, so they are never older than the rows
    # (db_restore.py also lifts every sequence to max(column) of the restored data)
    seq = client.json({"mode": "schema"})
    (work / "schema" / "sequences.sql").write_text(seq.get("sequences_sql") or "", encoding="utf-8")

    manifest = {
        "kind": "nics_trading_db_backup", "format": 1,
        "started_at": started.isoformat(), "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "server_schema_generated_at": schema.get("generated_at"),
        "tables": result, "errors": errors,
        "total_rows": sum(v["rows"] for v in result.values()),
        "total_file_bytes": sum(v["file_bytes"] for v in result.values()),
        "notes": ["app_secrets: secret values replaced by __REDACTED__ on the server",
                  "row order: primary key; tables without a primary key in physical order",
                  "rows written while the backup ran may or may not be included (no single snapshot)"],
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if errors:
        raise RuntimeError(f"{len(errors)} table(s) failed, backup kept as {work.name}: {sorted(errors)}")
    work.rename(final)
    prune(out_root, keep, log)
    log(f"OK {final}  rows={manifest['total_rows']}  size={manifest['total_file_bytes'] // (1024 * 1024)} MB")
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"backup root folder (default {DEFAULT_OUT})")
    ap.add_argument("--keep", type=int, default=14, help="complete backups to keep (default 14)")
    ap.add_argument("--chunk-mb", type=float, default=8.0, help="raw data per request, MB (default 8)")
    ap.add_argument("--tables", nargs="*", help="only these tables (for tests)")
    args = ap.parse_args(argv)

    url, key = backup_url(), backup_key()
    if not key:
        sys.exit("No NICS_BACKUP_KEY: n8n -> [00 OPS] Database Backup Export -> Create or Show Backup Key, "
                 "then run this script again and paste the key when asked.")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)

    try:
        final = run(Client(url, key), out_root, args.keep, int(args.chunk_mb * 1024 * 1024), args.tables, log)
    except RuntimeError as e:
        log(f"FAILED: {e}")
        if not args.tables:
            report(False, url, None, str(e), log)
        return 1
    if not args.tables:  # test runs with --tables are not real backups
        report(True, url, final, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
