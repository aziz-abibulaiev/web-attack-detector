"""Numeric features describing the shape of a request string: length, entropy, character-class
ratios, encoding depth, punctuation density. No feature here tests for a specific attack token.
"""

from __future__ import annotations

import math
from collections import Counter

from .normalize import assemble_fields, decode_fixed_point

_PUNCT_TOKENS = "'\"<>;{}()|&`"


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return float(-sum((k / n) * math.log2(k / n) for k in c.values()))


def _ratio(pred, s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if pred(ch)) / len(s)


def _nonalnum_ratio(s: str) -> float:
    return _ratio(lambda c: not c.isalnum(), s)


def _pct_encoded_ratio(s: str) -> float:
    if not s:
        return 0.0
    # count %XX occurrences relative to length: how much encoding, not what is encoded
    n = 0
    i = 0
    while i < len(s) - 2:
        if s[i] == "%" and s[i + 1] in "0123456789abcdefABCDEF" and s[i + 2] in "0123456789abcdefABCDEF":
            n += 1
            i += 3
        else:
            i += 1
    return n / max(1, len(s))


def _punct_density(s: str) -> float:
    if not s:
        return 0.0
    return sum(s.count(p) for p in _PUNCT_TOKENS) / len(s)


def _shape_block(name: str, s: str) -> dict:
    return {
        f"{name}_len": float(len(s)),
        f"{name}_entropy": _entropy(s),
        f"{name}_nonalnum_ratio": _nonalnum_ratio(s),
        f"{name}_digit_ratio": _ratio(str.isdigit, s),
        f"{name}_upper_ratio": _ratio(str.isupper, s),
        f"{name}_space_ratio": _ratio(str.isspace, s),
        f"{name}_punct_density": _punct_density(s),
    }


# Stable, declared feature order.
FEATURE_NAMES = (
    ["path_len", "query_len", "body_len", "total_len", "query_param_count",
     "path_depth", "segment_count", "encoding_depth", "pct_encoded_ratio"]
    + [f"{n}_{stat}" for n in ("path", "query", "body", "hdr")
       for stat in ("len", "entropy", "nonalnum_ratio", "digit_ratio", "upper_ratio",
                    "space_ratio", "punct_density")]
)


def numeric_features(*, url_path_raw="", url_query="", request_body="", headers=None) -> dict:
    """Compute the shape-feature dict for one request. Decodes to fixed point first, so the
    same encoding_depth and decoded shapes are seen in train and inference."""
    fields = assemble_fields(url_path_raw=url_path_raw, url_query=url_query,
                             request_body=request_body, headers=headers)
    path_dec, _ = decode_fixed_point(url_path_raw or "")
    query_dec, _ = decode_fixed_point(url_query or "")
    body_dec, _ = decode_fixed_point(request_body or "")
    hdr = fields["field_b"]

    feats = {
        "path_len": float(len(path_dec)),
        "query_len": float(len(query_dec)),
        "body_len": float(len(body_dec)),
        "total_len": float(len(fields["field_a"])),
        "query_param_count": float(query_dec.count("=") if query_dec else 0),
        "path_depth": float(sum(1 for seg in path_dec.split("/") if seg)),
        "segment_count": float(len([seg for seg in path_dec.split("/")])),
        "encoding_depth": float(fields["encoding_depth"]),
        "pct_encoded_ratio": _pct_encoded_ratio(fields["raw_target"]),
    }
    feats.update(_shape_block("path", path_dec))
    feats.update(_shape_block("query", query_dec))
    feats.update(_shape_block("body", body_dec))
    feats.update(_shape_block("hdr", hdr))
    return feats


def features_from_corpus_row(row) -> dict:
    return numeric_features(
        url_path_raw=row.get("url_path_raw", ""),
        url_query=row.get("url_query", ""),
        request_body=row.get("request_body", ""),
        headers=row.get("headers", ""),
    )
