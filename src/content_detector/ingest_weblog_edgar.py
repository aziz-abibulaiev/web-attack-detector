"""Reads the WebLog-2025 logs from Zenodo 10.5281/zenodo.20001206 and the SEC EDGAR logs into
the unified schema. EDGAR ships daily CSVs with no attack labels; the URI is reconstructed as
/Archives/edgar/data/{cik}/{accession}{extention}. WebLog labels come from the dataset's own
YAML pattern rules, which define the benign/attack partition and the recall set but never
filter training benign.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd


# ---------------- EDGAR ----------------
def parse_edgar(csv_paths, out, cap=None, nrows_per_file=None):
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p, usecols=["date", "time", "cik", "accession", "extention", "code"],
                         dtype=str, na_filter=False, nrows=nrows_per_file)
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    cik = pd.to_numeric(d["cik"], errors="coerce").fillna(0).astype("int64").astype(str)
    ext = d["extention"].fillna("")
    d["http_method"] = "GET"
    d["url_path_raw"] = "/Archives/edgar/data/" + cik + "/" + d["accession"].astype(str) + ext
    d["url_query"] = ""
    d["request_body"] = ""
    d["headers"] = ""
    d["timestamp"] = (d["date"].astype(str) + "T" + d["time"].astype(str) + ".000Z")
    d["label"] = "benign"
    d = d[["timestamp", "http_method", "url_path_raw", "url_query", "request_body", "headers", "label"]]
    sig = d.http_method + "|" + d.url_path_raw
    d = d.loc[~sig.duplicated()].copy()
    if cap and len(d) > cap:
        d = d.sample(n=cap, random_state=42)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    d.to_parquet(out, index=False, compression="zstd")
    print(f"[edgar] files={len(csv_paths)} unique_benign={len(d)} -> {out}")
    print(f"[edgar] sample URIs: {d.url_path_raw.head(3).tolist()}")


# ---------------- WebLog-2025 Org Y ----------------
# combined/OpenLiteSpeed: host - - [ts] "METHOD uri HTTP/x" status size "referer" "ua"
_COMBINED = re.compile(
    r'^(\S+) \S+ \S+ \[([^\]]+)\] "([A-Z]+) (\S+) [^"]*" (\d{3}) (\d+|-) "([^"]*)" "([^"]*)"')


def _load_yaml_rules(yaml_path):
    """Parse the dataset's ground-truth YAML into a list of label and filter-substring pairs.
    Filters use % as a LIKE wildcard, which is stripped to leave a case-insensitive substring,
    and every filter in a rule must match. Minimal parser, so PyYAML is not a dependency."""
    rules = []
    cur_label, cur_filters, in_filter = None, [], False
    for raw in open(yaml_path, encoding="utf-8"):
        line = raw.rstrip("\n")
        s = line.strip()
        if s.startswith("- id:"):
            if cur_label and cur_filters:
                rules.append((cur_label, cur_filters))
            cur_label, cur_filters, in_filter = None, [], False
        elif s.startswith("ground_truth_label:"):
            cur_label = s.split(":", 1)[1].strip()
        elif s.startswith("filter:"):
            in_filter = True
        elif in_filter and s.startswith("- "):
            v = s[2:].strip().strip('"').strip("'").replace("%", "")
            if v:
                cur_filters.append(v.lower())
    if cur_label and cur_filters:
        rules.append((cur_label, cur_filters))
    return rules


def parse_weblog(log_paths, out, yaml_path, cap=0):
    rules = _load_yaml_rules(yaml_path)
    print(f"[weblog] loaded {len(rules)} YAML ground-truth rules")
    from collections import Counter
    from datetime import datetime
    rows = []
    seen = set()          # dedup on method|path|query, keeping the first occurrence
    raw_labels = Counter()  # every request before dedup, to report the true attack volume
    parsed = 0
    unparsed = 0
    for path in log_paths:
        with open(path, "r", encoding="latin-1", errors="replace") as fh:
            for line in fh:
                m = _COMBINED.match(line)
                if not m:
                    unparsed += 1
                    continue
                parsed += 1
                host, ts, method, tgt, status, size, ref, ua = m.groups()
                up, uq = (tgt.split("?", 1) + [""])[:2]
                # apply the dataset's own YAML rules to the raw access-log line. This is the
                # dataset's ground truth, not our detector regex, and it defines this dataset's
                # labels only; it never filters training benign. label = rule id, or benign.
                low = line.lower()
                gt = "benign"
                for lab, filters in rules:
                    if all(f in low for f in filters):
                        gt = lab
                        break
                raw_labels[gt] += 1
                label = "benign" if gt == "benign" else "attack"
                sig = method + "|" + up + "|" + uq
                if sig in seen:
                    continue
                seen.add(sig)
                try:
                    tsi = datetime.strptime(ts.split()[0], "%d/%b/%Y:%H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S.000Z")
                except Exception:
                    tsi = ""
                rows.append((tsi, method, up, uq, json.dumps({"User-Agent": ua, "Referer": ref}), label, gt))
                if cap and parsed >= cap:
                    break
        if cap and parsed >= cap:
            break
    df = pd.DataFrame(rows, columns=["timestamp", "http_method", "url_path_raw", "url_query", "headers", "label", "gt_rule"])
    df["request_body"] = ""
    df = df[["timestamp", "http_method", "url_path_raw", "url_query", "request_body", "headers", "label", "gt_rule"]]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    vc = df.label.value_counts().to_dict()
    print(f"[weblog] parsed={parsed} unparsed={unparsed} unique={len(df)} unique_labels={vc}")
    print(f"[weblog] raw label volume before dedup: {dict(raw_labels.most_common())}")
    print(f"[weblog] unique gt_rule breakdown: {df.gt_rule.value_counts().to_dict()}")
    print(f"[weblog] sample attack paths: {df[df.label=='attack'].url_path_raw.head(5).tolist()} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgar-csv", nargs="*", default=[])
    ap.add_argument("--edgar-out")
    ap.add_argument("--edgar-cap", type=int, default=0)
    ap.add_argument("--edgar-nrows", type=int, default=0)
    ap.add_argument("--weblog", nargs="*", default=[])
    ap.add_argument("--weblog-out")
    ap.add_argument("--weblog-yaml")
    ap.add_argument("--weblog-cap", type=int, default=0)
    args = ap.parse_args()
    if args.edgar_csv:
        parse_edgar(args.edgar_csv, args.edgar_out, cap=args.edgar_cap or None,
                    nrows_per_file=args.edgar_nrows or None)
    if args.weblog:
        parse_weblog(args.weblog, args.weblog_out, args.weblog_yaml, cap=args.weblog_cap)


if __name__ == "__main__":
    main()
