"""Runs the same in-domain protocol on two deployment domains, zanbil and SR-BH: one text-only
model per domain, threshold at 1% FPR on that domain's own benign, recall from held-out attack
sources. Writes models/multidomain/multidomain_results.json, which backs Section 7.2.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import sources as SRC
from .normalize import decode_fixed_point
from .predict import _EXPLAIN_RULES
from .in_domain_train import TextOnly

MDIR = "models/multidomain"
EXT = "data/corpus/external"
TRAIN_BEN_CAP, CALIB_BEN_CAP, TEST_BEN_CAP = 120_000, 25_000, 100_000
SRBH_ATTACK_CAP = 60_000
HELDOUT_TOOL = "custom"
TEMPORAL_Q = 0.70

# WAMM inspection aid only, never a feature or a label. Applied to the decoded request so it
# catches encoded payloads (SR-BH stores {{ }} as %7B%7B, sleep(15) as %2815%29, etc.), and
# includes shellshock / sleep-injection / template markers seen in honeypot traffic.
_AUDIT = re.compile(
    r"(union|select\s|\bnull,null|sleep\s*\(|benchmark\(|waitfor|rlike|information_schema|"
    r"concat\(|char\(|\bor\s+1=1|'\s*or\s*'1'='1|<script|onerror|javascript:|\.\./|/etc/passwd|"
    r"\(\)\s*\{\s*:;\}|/bin/|\{\{|\$\{|<%=|;\s*(cat|ls|id|whoami)|\|\s*(id|whoami)|\$\(|`|%00|"
    r"169\.254\.169\.254|gopher://|/\.git|/\.env|wp-config|/phpinfo|cmd=|\bexec\b|passwd)", re.I)


def _marked(row):
    dec, _ = decode_fixed_point(f"{row.url_path_raw} {row.get('url_query','')} {row.get('request_body','')}")
    return bool(_AUDIT.search(dec))


def _sig(df):
    return (df.http_method.astype(str) + "|" + df.url_path_raw.astype(str) + "|"
            + df.url_query.astype(str) + "|" + df.get("request_body", pd.Series([""] * len(df))).astype(str))


def _reqcols(df):
    d = df.copy()
    for c in ["url_path_raw", "url_query", "request_body", "headers"]:
        if c not in d:
            d[c] = ""
    return d[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records")


def _batched_p(m, df, batch=40_000):
    out = []
    for i in range(0, len(df), batch):
        out.append(m.p(pd.DataFrame(_reqcols(df.iloc[i:i + batch]))))
    return np.concatenate(out) if out else np.array([])


def regex_recall(df):
    hits = 0
    for r in _reqcols(df):
        dec, _ = decode_fixed_point(f"{r['url_path_raw']} {r['url_query']} {r['request_body']}")
        hits += any(p.search(dec) for p in _EXPLAIN_RULES.values())
    return hits / len(df) if len(df) else 0.0


# ---------- shared attacks ----------
def build_attacks():
    srbh = SRC.load_srbh(); sa = srbh[srbh.label == "attack"]
    def tr(k):
        return int(hashlib.sha256(f"{C.SEED}:{k}".encode()).hexdigest(), 16) % 10 < 7
    m = _sig(sa).map(tr)
    srbh_tr = sa[m].sample(n=min(SRBH_ATTACK_CAP, int(m.sum())), random_state=C.SEED)
    srbh_te = sa[~m].sample(n=min(40_000, int((~m).sum())), random_state=C.SEED)
    tb = SRC.load_testbed_full(); ta = tb[tb.label == "attack"]
    tb_tr = ta[ta.attack_tool != HELDOUT_TOOL]; tb_te = ta[ta.attack_tool == HELDOUT_TOOL]
    assert not (set(_sig(srbh_tr)) & set(_sig(srbh_te))), "SR-BH attack train/test overlap"
    return dict(train=pd.concat([srbh_tr, tb_tr], ignore_index=True), srbh_te=srbh_te, custom_te=tb_te)


# ---------- benign in-domain split ----------
def split_benign(df, kind):
    df = df.loc[~_sig(df).duplicated()].copy()
    if kind == "temporal" and "timestamp" in df:
        df["ts"] = pd.to_datetime(df.timestamp, errors="coerce")
        df = df[df.ts.notna() & (df.ts.dt.year > 1970)]
        b = df.ts.quantile(TEMPORAL_Q)
        early = df[df.ts < b].sample(frac=1.0, random_state=C.SEED); late = df[df.ts >= b]
        method = f"temporal@{b}"
    else:
        h = _sig(df).map(lambda k: int(hashlib.sha256(f"{C.SEED}:{k}".encode()).hexdigest(), 16) % 100)
        early = df[h < 70].sample(frac=1.0, random_state=C.SEED); late = df[h >= 70]
        method = "hash70"
    # carve calib as a fraction of the early pool, which holds up on small sources; train is the rest
    n_calib = min(CALIB_BEN_CAP, max(1, int(len(early) * 0.15)))
    calib = early.iloc[:n_calib]
    train = early.iloc[n_calib:n_calib + TRAIN_BEN_CAP]
    test = late.sample(n=min(TEST_BEN_CAP, len(late)), random_state=C.SEED)
    assert not (set(_sig(train)) & set(_sig(calib))), "train/calib benign overlap"
    assert not (set(_sig(train)) & set(_sig(test))), "train/test benign overlap"
    assert not (set(_sig(calib)) & set(_sig(test))), "calib/test benign overlap"
    return train, calib, test, method


def run_domain(name, benign_df, split_kind, atk, modsec=None, xsource_fp=None):
    train_b, calib_b, test_b, method = split_benign(benign_df, split_kind)
    fit = pd.concat([train_b.assign(y=0), atk["train"].assign(y=1)], ignore_index=True)
    m = TextOnly().fit(pd.DataFrame(_reqcols(fit)), fit.y.values)
    os.makedirs(MDIR, exist_ok=True)
    joblib.dump(m, os.path.join(MDIR, f"text_only_{name}.joblib"))

    thr = {a: float(np.quantile(_batched_p(m, calib_b), 1 - a, method="higher")) for a in C.FPR_BUDGETS}
    p_late = _batched_p(m, test_b); p_srbh = _batched_p(m, atk["srbh_te"]); p_custom = _batched_p(m, atk["custom_te"])
    p_modsec = _batched_p(m, modsec) if modsec is not None else None

    res = {"split_method": method, "benign_train": int(len(train_b)), "benign_calib": int(len(calib_b)),
           "benign_test_late": int(len(test_b)), "attack_train": int(len(atk["train"]))}
    for a in C.FPR_BUDGETS:
        res[f"fpr_{a}"] = {
            "late_benign_alert": float((p_late >= thr[a]).mean()),
            "srbh_recall_in_dist": float((p_srbh >= thr[a]).mean()),
            "testbed_custom_cross_tool_recall": float((p_custom >= thr[a]).mean()),
            "modsec_recall": (float((p_modsec >= thr[a]).mean()) if p_modsec is not None else None),
        }
    # WAMM on top-100 late-benign alerts
    B = test_b.reset_index(drop=True).copy(); B["_p"] = p_late
    top = B[B["_p"] >= thr[C.PRIMARY_FPR]].sort_values("_p", ascending=False).head(C.MANUAL_INSPECT_N)
    marked = top.apply(_marked, axis=1) if len(top) else pd.Series([], dtype=bool)
    n_top = int(len(top)); n_mark = int(marked.sum()) if len(top) else 0
    res["wamm_manual"] = {"n_top": n_top, "regex_marked_plausible_attack": n_mark, "true_fp_est": n_top - n_mark}
    tf = (n_top - n_mark) / n_top if n_top else 1.0
    for a in C.FPR_BUDGETS:
        res[f"fpr_{a}"]["late_benign_fp_corrected"] = float(res[f"fpr_{a}"]["late_benign_alert"] * tf)
    res["cross_source_heldout_fp_reference"] = xsource_fp
    if len(top):
        top.assign(marked=marked.values)[["http_method", "url_path_raw", "url_query", "_p", "marked"]] \
            .to_csv(os.path.join(MDIR, f"manual_inspect_{name}.csv"), index=False)
    return res


def load_modsec():
    p = os.path.join(EXT, "modsec.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def main():
    os.makedirs(MDIR, exist_ok=True)
    atk = build_attacks()
    modsec = load_modsec()
    out = {"config": {"seed": C.SEED, "budgets": list(C.FPR_BUDGETS),
                      "modsec_available": modsec is not None,
                      "modsec_n": int(len(modsec)) if modsec is not None else 0}}

    # domain benigns
    zan = pd.read_parquet(f"{EXT}/zanbil.parquet",
                          columns=["timestamp", "http_method", "url_path_raw", "url_query", "headers", "label"])
    zan = zan[zan.label == "benign"]; zan["request_body"] = ""
    srb = pd.read_parquet(f"{EXT}/srbh.parquet",
                          columns=["timestamp", "http_method", "url_path_raw", "url_query", "request_body", "headers", "label"])
    srb = srb[srb.label == "benign"]

    domains = {"zanbil": (zan, "temporal", 0.995), "srbh": (srb, "temporal", 0.98)}

    out["domains"] = {}
    for name, (bdf, kind, xref) in domains.items():
        print(f"[domain] {name} ({kind}) n_benign={len(bdf)} ...", flush=True)
        out["domains"][name] = run_domain(name, bdf, kind, atk, modsec=modsec, xsource_fp=xref)

    with open(os.path.join(MDIR, "multidomain_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    _print(out)
    return out


def _print(o):
    print("\n===== MULTI-DOMAIN IN-DOMAIN VALIDATION at the 1% FPR budget =====")
    print(f"  ModSec available: {o['config']['modsec_available']} (n={o['config']['modsec_n']})")
    hdr = f"  {'domain':10s} {'split':22s} {'late-FP':>9s} {'corr':>7s} {'SRBH':>7s} {'custom':>7s} {'ModSec':>7s}"
    print(hdr)
    for name, r in o["domains"].items():
        b = r["fpr_0.01"]
        ms = f"{b['modsec_recall']*100:6.1f}%" if b["modsec_recall"] is not None else "   n/a"
        print(f"  {name:10s} {r['split_method'][:22]:22s} {b['late_benign_alert']*100:7.2f}% "
              f"{b['late_benign_fp_corrected']*100:6.2f}% {b['srbh_recall_in_dist']*100:6.1f}% "
              f"{b['testbed_custom_cross_tool_recall']*100:6.1f}% {ms}")
        w = r["wamm_manual"]
        print(f"             WAMM top-{w['n_top']}: real={w['regex_marked_plausible_attack']} "
              f"true-FP={w['true_fp_est']}  | cross-source-heldout-ref FP={r['cross_source_heldout_fp_reference']}")


if __name__ == "__main__":
    main()
