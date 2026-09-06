"""Reads the ModSecurity 2025 audit logs from Zenodo 10.5281/zenodo.17178461 into the unified
schema.
The logs are multipart; the -B-- section holds the request line and headers. Every logged
transaction tripped a CRS rule, so the output is attack-only.

Run:  python -m src.content_detector.ingest_modsec --src <dir-of-day-logs> --out data/corpus/external/modsec.parquet
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd

from . import config as C

_BOUND = re.compile(r"^--[0-9a-fA-F]+-([A-Z])--\s*$")
_REQLINE = re.compile(r"^([A-Z]+)\s+(\S+)\s+HTTP/[0-9.]+\s*$")


def parse_file(path, rows):
    mode = None
    buf = []
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            m = _BOUND.match(line)
            if m:
                if mode == "B" and buf:
                    _finish(buf, rows)
                mode = m.group(1)
                buf = []
                continue
            if mode == "B":
                buf.append(line.rstrip("\n"))


def _finish(buf, rows):
    if not buf:
        return
    rl = _REQLINE.match(buf[0])
    if not rl:
        return
    method, target = rl.group(1), rl.group(2)
    up, uq = (target.split("?", 1) + [""])[:2]
    # headers until blank line; body after
    headers = {}
    i = 1
    while i < len(buf) and buf[i].strip():
        if ":" in buf[i]:
            k, v = buf[i].split(":", 1)
            headers[k.strip()] = v.strip()
        i += 1
    body = "\n".join(buf[i + 1:]).strip() if i + 1 < len(buf) else ""
    rows.append({
        "http_method": method, "url_path_raw": up, "url_query": uq,
        "request_body": body, "headers": json.dumps(headers, ensure_ascii=False),
        "label": "attack", "source_id": "modsec", "nature": "real",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir containing */modsec_audit.anon.log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=40_000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*", "modsec_audit.anon.log")))
    rows = []
    for f in files:
        parse_file(f, rows)
        if len(rows) >= args.sample * 3:  # gather a surplus, then dedup + sample
            break
    df = pd.DataFrame(rows)
    sig = (df.http_method + "|" + df.url_path_raw + "|" + df.url_query + "|" + df.request_body)
    df = df.loc[~sig.duplicated()].copy()
    if len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=C.SEED)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")
    print(f"[modsec] files={len(files)} parsed_unique={len(df)} -> {args.out}")
    print(f"[modsec] method dist: {df.http_method.value_counts().to_dict()}")
    print(f"[modsec] sample paths: {df.url_path_raw.head(4).tolist()}")


if __name__ == "__main__":
    main()
