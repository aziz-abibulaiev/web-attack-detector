"""Loads real benign traffic, deduplicates it on the request signature and splits it with a
stable hash of that same key, so train and calib never share a row. Benign traffic is used as
it arrives and is never filtered with the detection regexes, so the attacks it already contains
stay in the training data, about 9-10% of SR-BH "benign". Keeping the training benign source
different from the false-positive test source is the caller's job.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from . import config as C

SRBH = "data/corpus/external/srbh.parquet"
ZANBIL = "data/corpus/external/zanbil.parquet"
_COLS = ["http_method", "url_path_raw", "url_query", "request_body", "headers", "label"]


def _sig(df: pd.DataFrame) -> pd.Series:
    return (df.http_method.astype(str) + "|" + df.url_path_raw.astype(str) + "|"
            + df.url_query.astype(str) + "|" + df.request_body.astype(str))


def _bucket(key: str) -> str:
    h = int(hashlib.sha256(f"{C.SEED}:{key}".encode()).hexdigest(), 16) % 10_000 / 10_000.0
    return "train" if h < C.REAL_BENIGN_TRAIN_FRAC else "calib"


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    sig = _sig(df)
    return df.loc[~sig.duplicated()].copy()


def srbh_benign_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_parquet(SRBH, columns=_COLS)
    b = dedup(s[s.label == "benign"])
    b["_b"] = _sig(b).map(_bucket)
    return b[b._b == "train"].drop(columns="_b"), b[b._b == "calib"].drop(columns="_b")


def srbh_benign_all() -> pd.DataFrame:
    """Full deduped SR-BH benign — for use as a held-out FPR-test source (A2)."""
    s = pd.read_parquet(SRBH, columns=_COLS)
    return dedup(s[s.label == "benign"])


def srbh_attacks(n: int | None = None) -> pd.DataFrame:
    s = pd.read_parquet(SRBH, columns=_COLS)
    a = dedup(s[s.label == "attack"])
    if n and len(a) > n:
        a = a.sample(n=n, random_state=C.SEED)
    return a


def zanbil_dedup(n_unique: int | None = None, exclude_keys: set | None = None) -> pd.DataFrame:
    """Deduped zanbil, optionally down to n_unique rows, optionally excluding given sig keys."""
    z = pd.read_parquet(ZANBIL, columns=[c for c in _COLS if c != "request_body"])
    z["request_body"] = ""
    z = dedup(z)
    if exclude_keys:
        z = z.loc[~_sig(z).isin(exclude_keys)]
    if n_unique and len(z) > n_unique:
        z = z.sample(n=n_unique, random_state=C.SEED)
    return z


def zanbil_split() -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """Sample deduped zanbil for A2 train/calib; returns train, calib and the used sig keys."""
    z = zanbil_dedup(n_unique=C.ZANBIL_TRAIN_SAMPLE)
    z["_b"] = _sig(z).map(_bucket)
    used = set(_sig(z))
    return z[z._b == "train"].drop(columns="_b"), z[z._b == "calib"].drop(columns="_b"), used


def sig_keys(df: pd.DataFrame) -> set:
    return set(_sig(df))
