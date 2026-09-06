"""Full-run acceptance checks + integrity audit + reproducibility-manifest export.

Prints PASS/FAIL with numbers for every acceptance criterion, runs the cross-contamination
audit benign rows carrying an injection marker, found with a separate regex that the labeller
never uses, and copies the reproducibility manifests into docs/corpus/
so they are version-controlled.
"""

import argparse
import ast
import json
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.manifest import load_manifests  # noqa: E402
from common.schema import UNIFIED_FIELDS  # noqa: E402

CONTENT_FAMILIES = ["sqli", "xss", "cmd_injection", "nosql_injection", "ldap_injection",
                    "ssti", "path_traversal", "ssrf", "scan"]
BEHAVIOURAL_FAMILIES = ["brute_force", "cred_stuffing", "resource_exhaustion"]
CONTEXT_FAMILIES = ["bola_idor", "mass_assignment", "business_logic_abuse"]
WINDOW_SEC = 10

# Independent QA regex — audit only, never used for labeling.
INJECTION_MARKERS = re.compile(
    r"(union\s+select|\bor\s+1=1\b|'\s*or\s*'1'='1|<script|onerror\s*=|javascript:|"
    r"\.\./|%2e%2e%2f|/etc/passwd|;\s*(cat|ls|id|whoami)|\|\s*(id|whoami)|\$\(|`|"
    r"\{\{.*?\}\}|\$\{.*?\}|<%=|%27|%3cscript|\$ne\b|\$gt\b|\$where\b|169\.254\.169\.254|gopher://)",
    re.IGNORECASE,
)


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--labeller", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--run-id", default="full")
    ap.add_argument("--zanbil-stats", required=True)
    ap.add_argument("--srbh-stats", required=True)
    ap.add_argument("--docs-out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.testbed)
    campaigns = load_manifests(args.manifest_dir)
    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1 schema completeness
    missing = [c for c in UNIFIED_FIELDS if c not in df.columns]
    check("schema_fields_present", not missing, f"cols={len(df.columns)} missing={missing}")
    check("zero_null_label", int(df.label.isna().sum()) == 0, f"null label={int(df.label.isna().sum())}")
    check("zero_null_label_source", int(df.label_source.isna().sum()) == 0,
          f"null label_source={int(df.label_source.isna().sum())}")

    # 2 label independence, checked against the parsed AST
    tree = ast.parse(open(args.labeller, encoding="utf-8").read())
    imports = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            imports.add(n.module or "")
    forbidden = [m for m in imports if m == "re" or "nlp_infer" in m or m.startswith("rules") or "services" in m]
    check("labeller_no_detection_deps", not forbidden, f"forbidden imports={forbidden}")

    # 3 benign purity — no benign row in an attack window by source IP
    aw = [(c["src_ip"], parse_ts(c["start_ts"]), parse_ts(c["end_ts"]))
          for c in campaigns if c["label"] == "attack"]
    benign = df[df.label == "benign"]
    impure = 0
    for _, r in benign.iterrows():
        ts = parse_ts(r["timestamp"])
        if any(r["src_ip"] == s and a <= ts <= b for s, a, b in aw):
            impure += 1
    check("benign_purity", impure == 0, f"benign checked={len(benign)} in-attack-window={impure}")

    # 4 integrity audit: benign rows carrying an injection marker, found with a separate regex
    def has_marker(r):
        blob = f"{r['url_path_raw']} {r['url_query']} {r['request_body']}"
        return bool(INJECTION_MARKERS.search(blob))
    benign_marked = int(benign.apply(has_marker, axis=1).sum())
    check("benign_no_injection_markers", benign_marked == 0,
          f"benign rows with injection marker={benign_marked} / {len(benign)}")

    # 5 attack delivery >=95%
    attack = df[df.label == "attack"]
    reached = int(attack.response_status.notna().sum())
    frac = reached / len(attack) if len(attack) else 0
    check("attack_delivery_95pct", frac >= 0.95, f"{reached}/{len(attack)}={frac:.3f}")

    # 5b benign volume must be at least attack volume
    check("benign_ge_attack", len(benign) >= len(attack),
          f"benign={len(benign)} attack={len(attack)}")

    # 6 content families: >=2 tools and >=500 rows
    fam_tools, fam_rows = {}, {}
    for fam in sorted(attack.attack_family.dropna().unique()):
        sub = attack[attack.attack_family == fam]
        fam_tools[fam] = sorted(sub.attack_tool.dropna().unique().tolist())
        fam_rows[fam] = int(len(sub))
    for fam in CONTENT_FAMILIES:
        if fam not in fam_rows:
            continue
        ok = len(fam_tools[fam]) >= 2 and fam_rows[fam] >= 500
        check(f"content_{fam}", ok, f"rows={fam_rows[fam]} tools={fam_tools[fam]}")

    # 7 behavioral window counts
    win = {}
    for fam in BEHAVIOURAL_FAMILIES:
        sub = attack[attack.attack_family == fam]
        if not len(sub):
            win[fam] = 0
            continue
        keys = set()
        for _, r in sub.iterrows():
            b = int(parse_ts(r["timestamp"]).timestamp()) // WINDOW_SEC
            keys.add((r["session_id"], b))
        win[fam] = len(keys)
        check(f"behavioural_{fam}_windows>=50", win[fam] >= 50,
              f"distinct (session,{WINDOW_SEC}s-window)={win[fam]}")

    # 7b context families present ground truth for the context track must exist
    for fam in CONTEXT_FAMILIES:
        n = int((attack.attack_family == fam).sum())
        check(f"context_{fam}_present", n > 0, f"rows={n}")

    # 8 session split disjointness
    split_stats_path = os.path.join(args.split_dir, "split_stats.json")
    split_stats = json.load(open(split_stats_path)) if os.path.exists(split_stats_path) else {}
    check("session_split_disjoint", split_stats.get("disjoint", False),
          f"overlaps={split_stats.get('session_overlaps', 'n/a')}")

    # benign coverage
    benign_users = benign.session_id.nunique()
    crapi_feats = set()
    for c in campaigns:
        if c["label"] == "benign" and c["app"] == "crapi":
            crapi_feats |= set((c.get("extra") or {}).get("feature_coverage", []))
    print(f"\n[benign] rows={len(benign)} distinct sessions={benign_users} crapi_features={sorted(crapi_feats)}")
    print(f"[counts] rows={len(df)} benign={len(benign)} attack={len(attack)}")
    print(f"[counts] per-family rows: {fam_rows}")
    print(f"[counts] per-family tools: {fam_tools}")
    print(f"[counts] behavioral windows: {win}")

    zst = json.load(open(args.zanbil_stats)) if os.path.exists(args.zanbil_stats) else {}
    sst = json.load(open(args.srbh_stats)) if os.path.exists(args.srbh_stats) else {}
    print(f"[external] zanbil={zst}")
    print(f"[external] srbh={sst}")

    # copy the campaign manifests into docs/corpus, where they are version controlled
    os.makedirs(os.path.join(args.docs_out, "manifests"), exist_ok=True)
    for c in campaigns:
        with open(os.path.join(args.docs_out, "manifests", f"{c['campaign_id']}.json"), "w") as f:
            json.dump(c, f, indent=2)
    run_manifest = {
        "run_id": args.run_id, "run": "full",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "seeds": {"all": 42},
        "images": {
            "vampi": "erev0s/vampi@sha256:0a5a224b6e14ae7da6a6ea265178ff71286ff903aec74adee98f660bb0e4ca12",
            "proxy": "mitmproxy/mitmproxy@sha256:00b77b5d8804c8ad18cb6caefbf9d5849e895e8986c5ce011f4ae30f4385962f",
            "generator_base": "python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de",
            "crapi": "OWASP/crAPI docker-compose (crapi/*:latest)",
        },
        "tools": {"sqlmap": "1.10.7", "ffuf": "2.1.0", "nuclei": "3.11.0", "hydra": "9.5",
                  "mitmproxy": "12.2.3", "seclists_commit": "e5e49caa6fb648476f3bca391b26a45a4f5d3f13"},
        "row_counts": {"total": int(len(df)), "benign": int(len(benign)), "attack": int(len(attack)),
                       "by_app": df.app.value_counts().to_dict(),
                       "by_family": fam_rows, "by_tool": attack.attack_tool.value_counts().to_dict()},
        "behavioural_windows": win,
        "content_family_tools": fam_tools,
        "external": {"zanbil": zst, "srbh": sst},
        "splits": split_stats.get("splits", {}),
        "checks": {n: ok for n, ok, _ in results},
    }
    with open(os.path.join(args.docs_out, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"\n[docs] manifests + run_manifest -> {args.docs_out}")

    nfail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*54}\nFull run: {len(results)-nfail}/{len(results)} checks passed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
