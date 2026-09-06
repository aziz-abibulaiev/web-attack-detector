"""Smoke-run acceptance checks — end-to-end validation of the small run.

Each check prints PASS/FAIL with numbers. Also writes the deterministic run manifest.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.manifest import load_manifests  # noqa: E402
from common.schema import UNIFIED_FIELDS  # noqa: E402


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--labeller", required=True)
    ap.add_argument("--run-id", default="smoke")
    ap.add_argument("--manifest-out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    campaigns = load_manifests(args.manifest_dir)
    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1 schema completeness
    missing_cols = [c for c in UNIFIED_FIELDS if c not in df.columns]
    check("schema_fields_present", not missing_cols,
          f"{len(df.columns)} cols, missing={missing_cols}")
    null_label = int(df["label"].isna().sum())
    null_src = int(df["label_source"].isna().sum())
    check("zero_null_label", null_label == 0, f"null labels={null_label}")
    check("zero_null_label_source", null_src == 0, f"null label_source={null_src}")

    # 2 label independence: inspect the labeller's real imports in the parsed AST, not prose.
    # Its docstring names rules/ and nlp_infer to say it avoids them, so a text grep would
    # report a false positive. Check the actual import statements and any use of `re`.
    import ast
    tree = ast.parse(open(args.labeller, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden_pkgs = [m for m in imported
                      if m == "re" or "nlp_infer" in m or m.startswith("rules")
                      or "base_rules" in m or "services" in m]
    check("labeller_no_detection_deps", not forbidden_pkgs,
          f"imports={sorted(imported)}; forbidden={forbidden_pkgs}")

    # 3 benign purity — no benign row inside any attack window
    attack_windows = [(c["src_ip"], parse_ts(c["start_ts"]), parse_ts(c["end_ts"]))
                      for c in campaigns if c["label"] == "attack"]
    benign = df[df["label"] == "benign"]
    impure = 0
    for _, r in benign.iterrows():
        ts = parse_ts(r["timestamp"])
        for sip, s, e in attack_windows:
            if r["src_ip"] == sip and s <= ts <= e:
                impure += 1
                break
    check("benign_purity", impure == 0,
          f"benign rows checked={len(benign)}, inside attack window={impure}")

    # 4 attack delivery — >=95% of attack rows reached the app
    attack = df[df["label"] == "attack"]
    if len(attack):
        reached = int(attack["response_status"].notna().sum())
        frac = reached / len(attack)
        check("attack_delivery_95pct", frac >= 0.95,
              f"{reached}/{len(attack)} reached app = {frac:.3f}")
    else:
        check("attack_delivery_95pct", False, "no attack rows")

    # 5 session_id present and sane
    null_sess = int(df["session_id"].isna().sum() + (df["session_id"] == "").sum())
    check("session_id_present", null_sess == 0, f"empty session_id={null_sess}")

    # 6 provenance — every row's label_source names its campaign
    bad_ls = int((~df.apply(lambda r: r["campaign_id"] in (r["label_source"] or ""), axis=1)).sum())
    check("label_source_from_manifest", bad_ls == 0, f"rows w/o campaign in label_source={bad_ls}")

    # per-family / per-tool counts
    fam = df[df["label"] == "attack"]["attack_family"].value_counts().to_dict()
    tool = df[df["label"] == "attack"]["attack_tool"].value_counts().to_dict()
    sess = df["session_id"].nunique()
    print(f"\n[counts] rows={len(df)} benign={int((df.label=='benign').sum())} "
          f"attack={int((df.label=='attack').sum())}")
    print(f"[counts] families={fam}")
    print(f"[counts] tools={tool}")
    print(f"[counts] distinct session_id={sess}")

    # run manifest
    run_manifest = {
        "run_id": args.run_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "run": "smoke",
        "seeds": {"benign": 42, "sqlmap": 42, "capture_run": args.run_id},
        "images": {
            "vampi": "erev0s/vampi@sha256:0a5a224b6e14ae7da6a6ea265178ff71286ff903aec74adee98f660bb0e4ca12",
            "proxy": "mitmproxy/mitmproxy@sha256:00b77b5d8804c8ad18cb6caefbf9d5849e895e8986c5ce011f4ae30f4385962f",
            "generator_base": "python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de",
        },
        "tools": {"sqlmap": "1.10.7", "mitmproxy": "12.2.3", "requests": "2.32.3", "PyJWT": "2.9.0"},
        "campaigns": campaigns,
        "row_counts": {
            "total": int(len(df)),
            "benign": int((df.label == "benign").sum()),
            "attack": int((df.label == "attack").sum()),
            "by_app": df["app"].value_counts().to_dict(),
            "by_family": fam,
            "by_tool": tool,
        },
        "checks": {name: ok for name, ok, _ in results},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)), exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as fh:
        json.dump(run_manifest, fh, indent=2)
    print(f"\n[manifest] wrote {args.manifest_out}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*50}\nSmoke run: {len(results)-n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
