"""Reads the zanbil.ir nginx access logs, real e-commerce traffic from Kaggle
eliasdabbas/web-server-access-logs, into the unified schema with app=zanbil, label=benign and
an empty request_body, since access logs carry no body. Any alert rate measured on this
traffic is an upper bound on the false-positive rate, because wild traffic contains real
attacks. Reports parsed rows against source lines and the parse-error rate, and streams the
3.3 GB file in chunks rather than loading it whole.
"""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import PARQUET_DTYPES, UNIFIED_FIELDS  # noqa: E402

_NS = uuid.UUID("6f1a7b2c-0000-4000-8000-00000000za00".replace("z", "0"))

# nginx combined-ish: IP - - [ts] "METHOD path HTTP/x.y" status size "referer" "ua"
LINE_RE = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<method>[A-Z]+)\s+(?P<path>[^"\s]+)\s+HTTP/(?P<ver>[0-9.]+)"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\d+|-)\s+'
    r'"(?P<referer>[^"]*)"\s+"(?P<ua>[^"]*)"'
)


def to_iso(ts: str) -> str:
    # e.g. 22/Jan/2019:03:56:14 +0330
    dt = datetime.strptime(ts.split()[0], "%d/%b/%Y:%H:%M:%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def build_schema() -> pa.Schema:
    return pa.schema([pa.field(c, pa.int64() if PARQUET_DTYPES[c] == "Int64"
                               else (pa.float64() if PARQUET_DTYPES[c] == "float64" else pa.string()))
                      for c in UNIFIED_FIELDS])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/raw/zanbil/access.log",
                    help="raw access log, as fetched by scripts/fetch_data.sh zanbil")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"[zanbil] source not found: {args.log}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    schema = build_schema()
    total = parsed = errors = 0
    buf = []
    writer = pq.ParquetWriter(args.out, schema, compression="zstd")

    def flush():
        nonlocal buf
        if not buf:
            return
        df = pd.DataFrame(buf, columns=UNIFIED_FIELDS)
        for c, dt in PARQUET_DTYPES.items():
            df[c] = df[c].astype(dt) if dt != "Int64" else pd.array(df[c], dtype="Int64")
        writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
        buf = []

    with open(args.log, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += 1
            m = LINE_RE.match(line)
            if not m:
                errors += 1
                continue
            g = m.groupdict()
            path = g["path"]
            up, uq = (path.split("?", 1) + [""])[:2]
            from urllib.parse import unquote
            buf.append({
                "capture_id": str(uuid.uuid5(_NS, f"zanbil:{total}")),
                "timestamp": to_iso(g["ts"]), "src_ip": g["ip"], "src_port": 0,
                "http_method": g["method"], "url_path": unquote(up), "url_path_raw": up,
                "url_query": uq, "http_version": g["ver"], "host": "zanbil.ir",
                "headers": json.dumps({"User-Agent": g["ua"], "Referer": g["referer"]}),
                "request_body": "", "request_content_type": "",
                "response_status": int(g["status"]),
                "response_size": 0 if g["size"] == "-" else int(g["size"]),
                "response_time_ms": None, "app": "zanbil",
                "session_id": f"zanbil:ip:{g['ip']}", "label": "benign",
                "attack_family": None, "attack_tool": None, "campaign_id": "zanbil_external",
                "label_source": "external=zanbil.ir nginx access log; organic benign, upper-bound FPR set",
            })
            parsed += 1
            if len(buf) >= args.chunk:
                flush()
            if args.max_rows and parsed >= args.max_rows:
                break
    flush()
    writer.close()

    rate = errors / total if total else 0.0
    stats = {"source_lines": total, "parsed_rows": parsed, "parse_errors": errors,
             "parse_error_rate": round(rate, 6), "out": args.out}
    print(json.dumps(stats, indent=2))
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
