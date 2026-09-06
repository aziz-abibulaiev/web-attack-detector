"""Assigns labels from campaign manifests alone.

Reads the raw unlabeled JSONL the capture addon wrote together with the campaign manifests,
and writes the labeled parquet in the unified schema.

The rule in full: a captured flow takes label, attack_family, attack_tool, campaign_id and
label_source from the single campaign whose source IP equals the flow's and whose
[start_ts, end_ts) window contains the flow's timestamp. Nothing else. This file imports no
detection rule or regex and never inspects request_body or url to decide a label.

It does read the Authorization header, but only to derive session_id: the JWT `sub` claim
from a Bearer token when one is present, decoded and not verified, since this is grouping
and not authentication, and otherwise the source IP.
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.manifest import load_manifests  # noqa: E402
from common.schema import PARQUET_DTYPES, UNIFIED_FIELDS  # noqa: E402


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")


def jwt_sub(auth_header: str) -> str | None:
    """Decode a Bearer JWT's `sub` claim WITHOUT verifying the signature.

    Session attribution only. A malformed or non-JWT token yields None.
    """
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header.split(None, 1)[1].strip()
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None
    sub = payload.get("sub")
    if isinstance(sub, dict):  # some encoders wrap {"username": ...}
        sub = sub.get("username") or sub.get("user")
    return str(sub) if sub is not None else None


def derive_session_id(row: dict, app: str) -> str:
    """Behavioral grouping key. API apps -> authenticated principal, else src_ip."""
    if app in ("vampi", "crapi"):
        try:
            headers = json.loads(row.get("headers") or "{}")
        except Exception:
            headers = {}
        auth = headers.get("Authorization") or headers.get("authorization") or ""
        sub = jwt_sub(auth)
        if sub:
            return f"{app}:user:{sub}"
    return f"{app}:ip:{row.get('src_ip', '')}"


def match_campaign(row: dict, campaigns: list[dict]):
    ts = parse_ts(row["timestamp"])
    for c in campaigns:
        if row["src_ip"] != c["src_ip"]:
            continue
        if parse_ts(c["start_ts"]) <= ts <= parse_ts(c["end_ts"]):
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="raw JSONL from the addon")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out", required=True, help="labelled parquet output")
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="drop flows matching no campaign instead of failing")
    args = ap.parse_args()

    campaigns = load_manifests(args.manifest_dir)
    if not campaigns:
        print("[label] no manifests found", file=sys.stderr)
        return 2

    rows = []
    with open(args.capture, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[label] {len(rows)} captured flows, {len(campaigns)} campaigns")

    labelled = []
    unmatched = 0
    for r in rows:
        c = match_campaign(r, campaigns)
        if c is None:
            unmatched += 1
            continue
        app = r.get("app") or c["app"]
        session_id = derive_session_id(r, app)
        window = f"[{c['start_ts']},{c['end_ts']})"
        label_source = (
            f"campaign={c['campaign_id']} tool={c['attack_tool']} "
            f"src={c['src_ip']} window={window}"
        )
        out = {k: r.get(k) for k in UNIFIED_FIELDS if k in r}
        out.update({
            "app": app,
            "session_id": session_id,
            "label": c["label"],
            "attack_family": c.get("attack_family"),
            "attack_tool": c.get("attack_tool"),
            "campaign_id": c["campaign_id"],
            "label_source": label_source,
        })
        labelled.append(out)

    print(f"[label] matched {len(labelled)}, unmatched {unmatched}")
    if unmatched and not args.allow_unmatched:
        print(f"[label] FAIL: {unmatched} flows matched no campaign window; "
              f"every flow must be attributable. Use --allow-unmatched to drop.", file=sys.stderr)
        return 3

    df = pd.DataFrame(labelled, columns=UNIFIED_FIELDS)
    for col, dt in PARQUET_DTYPES.items():
        try:
            df[col] = df[col].astype(dt)
        except (TypeError, ValueError):
            df[col] = pd.array(df[col], dtype=dt)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")
    print(f"[label] wrote {len(df)} rows -> {args.out}")
    print(df["label"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
