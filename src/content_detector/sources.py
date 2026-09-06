"""One loader per dataset. Each returns a DataFrame with the columns the ContentDetector reads
url_path_raw, url_query, request_body, headers and label, plus source_id, nature and has_body,
deduplicated within the source on the request signature. Benign traffic is loaded unfiltered,
and attack labels come only from a dataset's own labeling.
"""

from __future__ import annotations

import gzip
import os
import re
from urllib.parse import urlparse

import pandas as pd


_ROOT = "data/corpus/external"
_TESTBED = "data/corpus/labelled/testbed.parquet"
_CSIC = os.environ.get("CORPUS_CSIC_CSV", "data/raw/csic/csic_database.csv")
# ITA access logs, fetched separately for benign diversity. Override with CORPUS_ITA_DIR.
# Re-fetch: curl -L https://ita.ee.lbl.gov/traces/{NASA_access_log_Jul95,clarknet_access_log_Sep4,
#   calgary_access_log}.gz  -> nasa.gz / clark.gz / calgary.gz  — large, and gitignored.
_ITA = os.environ.get("CORPUS_ITA_DIR", "data/corpus/external/ita")

_UNI = ["http_method", "url_path_raw", "url_query", "request_body", "headers", "label"]
_CLF = re.compile(r'"(?P<m>[A-Z]+)\s+(?P<path>\S+)\s+HTTP/[0-9.]+"')


def _sig(df):
    return (df.http_method.astype(str) + "|" + df.url_path_raw.astype(str) + "|"
            + df.url_query.astype(str) + "|" + df.request_body.astype(str))


def _dedup(df):
    return df.loc[~_sig(df).duplicated()].copy()


def _finish(df, source_id, nature, has_body):
    for c in _UNI:
        if c not in df.columns:
            df[c] = ""
    df = df[_UNI].copy()
    df["request_body"] = df["request_body"].fillna("").astype(str)
    df = _dedup(df)
    df["source_id"] = source_id
    df["nature"] = nature
    df["has_body"] = has_body
    return df


# ---- unified-schema parquet sources ----
def load_zanbil():
    z = pd.read_parquet(f"{_ROOT}/zanbil.parquet",
                        columns=["http_method", "url_path_raw", "url_query", "headers", "label"])
    z["request_body"] = ""
    return _finish(z, "zanbil", "real", False)


def load_srbh():
    s = pd.read_parquet(f"{_ROOT}/srbh.parquet", columns=_UNI)
    return _finish(s, "srbh", "real", True)


def load_testbed_full():
    """testbed keeping attack_tool for tool-level holdout."""
    t = pd.read_parquet(_TESTBED, columns=_UNI + ["attack_tool"])
    t["request_body"] = t["request_body"].fillna("").astype(str)
    t = t.loc[~_sig(t).duplicated()].copy()
    t["source_id"] = "testbed"; t["nature"] = "testbed"; t["has_body"] = True
    return t


# ---- CSIC: semi-synthetic, both classes, request bodies present ----
def load_csic():
    if not os.path.exists(_CSIC):
        return None
    d = pd.read_csv(_CSIC, low_memory=False)
    # URL column carries a trailing " HTTP/1.1"; strip it, then parse path/query.
    url = d["URL"].astype(str).map(lambda u: u.split(" ")[0])
    parsed = url.map(lambda u: urlparse(u if u.startswith("http") else "http://x/" + u.lstrip("/")))
    import json as _json
    df = pd.DataFrame({
        "http_method": d["Method"].astype(str),
        "url_path_raw": parsed.map(lambda p: p.path),
        "url_query": parsed.map(lambda p: p.query),
        "request_body": d.get("content", "").astype(str) if "content" in d else "",
        "headers": d.get("User-Agent", pd.Series([""] * len(d))).map(
            lambda ua: _json.dumps({"User-Agent": str(ua)})),
        # classification is an integer: 0 = Normal is benign, 1 = Anomalous is attack
        "label": d["classification"].map(lambda c: "benign" if int(c) == 0 else "attack"),
    })
    return _finish(df, "csic", "semi-synthetic", True)


# ---- Internet Traffic Archive access logs: real benign, no request body ----
def _load_ita(fname, source_id, cap=None):
    path = os.path.join(_ITA, fname)
    if not os.path.exists(path):
        return None
    rows = []
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for line in fh:
            m = _CLF.search(line)
            if not m:
                continue
            raw = m.group("path")
            if not raw.startswith("/"):
                raw = "/" + raw
            up, uq = (raw.split("?", 1) + [""])[:2]
            rows.append((m.group("m"), up, uq))
            if cap and len(rows) >= cap:
                break
    df = pd.DataFrame(rows, columns=["http_method", "url_path_raw", "url_query"])
    df["request_body"] = ""; df["headers"] = ""; df["label"] = "benign"
    return _finish(df, source_id, "real", False)


def load_nasa(cap=400_000):
    return _load_ita("nasa.gz", "nasa", cap)


def load_clarknet(cap=400_000):
    return _load_ita("clark.gz", "clarknet", cap)


def load_calgary(cap=400_000):
    return _load_ita("calgary.gz", "calgary", cap)


REGISTRY = {
    "zanbil": load_zanbil, "srbh": load_srbh, "testbed": load_testbed_full,
    "csic": load_csic, "nasa": load_nasa, "clarknet": load_clarknet, "calgary": load_calgary,
}
