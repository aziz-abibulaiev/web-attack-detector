"""Decoding and field assembly, imported by both training and scoring so the two sides see the
same text. decode_fixed_point applies unquote_plus until the string stops changing, capped at
config.DECODE_MAX_PASSES.
"""

from __future__ import annotations

import json
from urllib.parse import unquote_plus

from .config import DECODE_MAX_PASSES


def decode_fixed_point(s: str, max_passes: int = DECODE_MAX_PASSES) -> tuple[str, int]:
    """Apply unquote_plus repeatedly until the string stops changing or the cap is hit.

    Returns the decoded string and the number of passes used, which is the encoding_depth
    feature. The repeat catches attacks that still hold a %XX escape after one pass.
    """
    if not s:
        return "", 0
    cur = s
    passes = 0
    for _ in range(max_passes):
        nxt = unquote_plus(cur)
        if nxt == cur:
            break
        cur = nxt
        passes += 1
    return cur, passes


def _clean(v) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        v = str(v)
    return v


def headers_get(headers, *names) -> str:
    """Case-insensitive header lookup. `headers` may be a dict or a JSON string."""
    if isinstance(headers, str):
        try:
            headers = json.loads(headers) if headers else {}
        except Exception:
            headers = {}
    if not isinstance(headers, dict):
        return ""
    low = {str(k).lower(): _clean(val) for k, val in headers.items()}
    for n in names:
        if n.lower() in low and low[n.lower()]:
            return low[n.lower()]
    return ""


def assemble_fields(*, url_path_raw="", url_query="", request_body="", headers=None) -> dict:
    """Build the two decoded text fields and carry the raw request-target string.

    field_a: the request target and body, where injection payloads live
    field_b: the user-agent and referer, a headers channel weighted separately
    raw_target: the undecoded url_path_raw+query+body, used for encoding_depth
    """
    raw_target = " ".join([_clean(url_path_raw), _clean(url_query), _clean(request_body)]).strip()
    ua = headers_get(headers, "user-agent")
    ref = headers_get(headers, "referer", "referrer")
    field_a, depth_a = decode_fixed_point(raw_target)
    field_b, depth_b = decode_fixed_point(" ".join([ua, ref]).strip())
    return {
        "field_a": field_a,
        "field_b": field_b,
        "raw_target": raw_target,
        "encoding_depth": max(depth_a, depth_b),
    }


def request_from_corpus_row(row) -> dict:
    """Adapt a dict-like corpus parquet row to the request shape both channels consume."""
    return assemble_fields(
        url_path_raw=row.get("url_path_raw", ""),
        url_query=row.get("url_query", ""),
        request_body=row.get("request_body", ""),
        headers=row.get("headers", ""),
    )
