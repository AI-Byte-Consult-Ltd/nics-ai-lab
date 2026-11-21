"""Offline test of scripts/db_backup.py and scripts/db_restore.py against a fake server.

The fake server speaks the same protocol as the n8n workflow "[00 OPS] Database Backup
Export" (modes tables / schema / chunk, gzip body, X-Rows / X-Last-Key headers, 403 on a
wrong key). Checks: keyset paging over a composite primary key, offset paging for a table
without one, empty tables, the .partial -> final rename, pruning old backups, and that the
restore SQL contains every row. No network, no database.

  python tests/test_db_backup.py
"""
import base64
import gzip
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import db_backup  # noqa: E402
import db_restore  # noqa: E402

KEY = "k" * 40
TABLES = {
    "candles": {"pk": ["symbol", "interval", "candle_time"],
                "rows": [{"symbol": s, "interval": i, "candle_time": f"2026-01-01T00:{m:02d}:00+00:00", "close": m}
                         for s in ("US500", "XAUUSD") for i in ("15min", "1min") for m in range(37)]},
    "nopk": {"pk": [], "rows": [{"a": n} for n in range(250)]},
    "empty_t": {"pk": ["id"], "rows": []},
    "app_secrets": {"pk": ["key"], "rows": [{"key": "telegram_bot_token", "value": "__REDACTED__"}]},
}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.headers.get("X-NICS-Backup-Key") != KEY:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"forbidden"}')
            return
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        rows, last = 0, ""
        if q["mode"] == "tables":
            body = {"ok": True, "tables": [
                {"table": n, "bytes": 1000 * len(t["rows"]), "est_rows": len(t["rows"]) or -1,
                 "pk": [{"col": c, "type": "text"} for c in t["pk"]] or None} for n, t in TABLES.items()]}
            data = json.dumps(body).encode()
        elif q["mode"] == "schema":
            data = json.dumps({"ok": True, "pre_data": "CREATE TABLE candles ();", "post_data": "-- fk",
                               "sequences_sql": "", "column_flags": [{"table": "candles", "generated": [],
                                                                      "identity_always": []}]}).encode()
        else:
            t = TABLES[q["table"]]
            limit = int(q["limit"])
            src = sorted(t["rows"], key=lambda r: [r[c] for c in t["pk"]]) if t["pk"] else t["rows"]
            if t["pk"] and q.get("after"):
                after = json.loads(base64.b64decode(q["after"]))
                src = [r for r in src if [r[c] for c in t["pk"]] > after]
            if not t["pk"]:
                src = src[int(q.get("offset", 0)):]
            page = src[:limit]
            rows = len(page)
            if page and t["pk"]:
                last = base64.b64encode(json.dumps([page[-1][c] for c in t["pk"]]).encode()).decode()
            data = "\n".join(json.dumps(r) for r in page).encode()
        self.send_response(200)
        self.send_header("X-Rows", str(rows))
        self.send_header("X-Last-Key", last)
        self.end_headers()
        self.wfile.write(gzip.compress(data))


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/webhook/{db_backup.WEBHOOK_PATH}"

    assert db_backup.backup_url({"NICS_DATA_URL": "https://h/webhook/nics-lab-market-export"}) == \
        "https://h/webhook/nics-db-backup"
    assert db_backup.backup_url({}) == db_backup.DEFAULT_URL

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "2020-01-01_0000").mkdir()
        (out / "2020-01-01_0000" / "manifest.json").write_text("{}")
        client = db_backup.Client(url, KEY, retries=1)
        client.get = _small_pages(client.get)
        final = db_backup.run(client, out, keep=1, target_bytes=10_000, only=None, log=lambda m: None)
        assert final.exists() and not list(out.glob("*.partial"))
        assert not (out / "2020-01-01_0000").exists(), "old backup should be pruned"
        man = json.loads((final / "manifest.json").read_text())
        for name, t in TABLES.items():
            got = [json.loads(line) for line in gzip.open(final / "data" / f"{name}.jsonl.gz", "rt") if line.strip()]
            assert len(got) == len(t["rows"]) == man["tables"][name]["rows"], name
            assert sorted(map(json.dumps, got)) == sorted(map(json.dumps, t["rows"])), name
        assert man["tables"]["candles"]["chunks"] > 3, "candles must have been paged"

        sql_path = out / "restore.sql"
        with open(sql_path, "w", encoding="utf-8") as f:
            counts = db_restore.build(final, f, batch=50)
        sql = sql_path.read_text(encoding="utf-8")
        assert counts["candles"] == len(TABLES["candles"]["rows"])
        assert sql.count('INSERT INTO public."candles"') == 3  # 148 rows / 50
        assert "json_populate_recordset" in sql and sql.index("CREATE TABLE") < sql.index("INSERT")

        bad = db_backup.Client(url, "wrong", retries=1)
        try:
            bad.get({"mode": "tables"})
            raise AssertionError("wrong key must stop the run")
        except SystemExit as e:
            assert "403" in str(e)
    srv.shutdown()
    print("db_backup / db_restore: OK")


def _small_pages(get):
    """Force tiny pages so paging is exercised (the server honours 'limit')."""
    def wrapped(params):
        if params.get("mode") == "chunk":
            params = dict(params, limit=40)
        return get(params)
    return wrapped


if __name__ == "__main__":
    main()
