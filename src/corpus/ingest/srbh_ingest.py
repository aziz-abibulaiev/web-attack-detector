"""Reads the SR-BH 2020 CSV and maps its raw HTTP fields into the unified schema, with
app=srbh. Only the binary label is taken: benign when `000 - Normal` is 1, otherwise attack.
The 13 CAPEC columns are not used as family ground truth, because they disagree with the
request content, so attack_family stays null.
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import PARQUET_DTYPES, UNIFIED_FIELDS  # noqa: E402

_NS = uuid.UUID("6f1a7b2c-0000-4000-8000-000000000db0")
NORMAL_COL = "000 - Normal"
HEADER_COLS = ["request_user_agent", "request_referer", "request_host", "request_origin",
               "request_cookie", "request_content_type", "request_accept",
               "request_accept_language", "request_accept_encoding", "request_do_not_track",
               "request_connection"]
WIRE = {c: c.replace("request_", "").replace("_", "-") for c in HEADER_COLS}


def to_iso(ts: str) -> str:
    try:
        dt = datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z").astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    except Exception:
        return "1970-01-01T00:00:00.000Z"


def build_schema() -> pa.Schema:
    return pa.schema([pa.field(c, pa.int64() if PARQUET_DTYPES[c] == "Int64"
                               else (pa.float64() if PARQUET_DTYPES[c] == "float64" else pa.string()))
                      for c in UNIFIED_FIELDS])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/srbh/data_capec_multilabel.csv",
                    help="SR-BH multilabel csv, as fetched by scripts/fetch_data.sh srbh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"[srbh] source not found: {args.csv}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    schema = build_schema()
    writer = pq.ParquetWriter(args.out, schema, compression="zstd")
    usecols = ["timestamp", "src_ip", "src_port", "request_http_method", "request_http_request",
               "request_http_protocol", "request_body", "response_http_status_code",
               "response_content_length", NORMAL_COL] + HEADER_COLS
    total = 0
    nb = na = 0
    seq = 0
    from urllib.parse import unquote

    for chunk in pd.read_csv(args.csv, usecols=usecols, dtype=str, na_filter=False, chunksize=args.chunk):
        rows = []
        # column-wise access: itertuples mangles non-identifier names like "000 - Normal"
        cols = {c: chunk[c].tolist() for c in usecols}
        for i in range(len(chunk)):
            d = {c: cols[c][i] for c in usecols}
            uri = d["request_http_request"]
            up, uq = (uri.split("?", 1) + [""])[:2]
            headers = {WIRE[c]: d[c] for c in HEADER_COLS if d[c]}
            normal = d[NORMAL_COL] == "1"
            label = "benign" if normal else "attack"
            nb += normal
            na += (not normal)
            status = d["response_http_status_code"]
            size = d["response_content_length"]
            rows.append({
                "capture_id": str(uuid.uuid5(_NS, f"srbh:{seq}")), "timestamp": to_iso(d["timestamp"]),
                "src_ip": d["src_ip"], "src_port": int(d["src_port"]) if d["src_port"].isdigit() else 0,
                "http_method": d["request_http_method"], "url_path": unquote(up), "url_path_raw": up,
                "url_query": uq, "http_version": d["request_http_protocol"],
                "host": d["request_host"], "headers": json.dumps(headers, ensure_ascii=False),
                "request_body": d["request_body"], "request_content_type": d["request_content_type"],
                "response_status": int(status) if status.isdigit() else None,
                "response_size": int(size) if size.isdigit() else 0, "response_time_ms": None,
                "app": "srbh", "session_id": f"srbh:ip:{d['src_ip']}", "label": label,
                "attack_family": None, "attack_tool": None, "campaign_id": "srbh_external",
                "label_source": "external=SR-BH 2020 honeypot; binary label from '000 - Normal'; CAPEC classes not used as family",
            })
            seq += 1
        df = pd.DataFrame(rows, columns=UNIFIED_FIELDS)
        for c, dt in PARQUET_DTYPES.items():
            df[c] = df[c].astype(dt) if dt != "Int64" else pd.array(df[c], dtype="Int64")
        writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
        total += len(rows)
    writer.close()

    stats = {"rows": total, "benign": int(nb), "attack": int(na), "out": args.out}
    print(json.dumps(stats, indent=2))
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
