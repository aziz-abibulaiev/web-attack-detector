"""Trains on several benign and attack sources at once and holds out entire sources per fold:
one benign source gives the alert rate, one attack source or tool gives recall, both at 1% and
5% FPR budgets set on a calibration slice from neither held-out source. Writes
models/loso/loso_results.json, which backs Section 7.1.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from . import config as C
from . import sources as SRC
from .model import ContentDetector

_AUDIT = re.compile(
    r"(union\s+select|\bor\s+1=1\b|'\s*or\s*'1'='1|<script|onerror\s*=|javascript:|\.\./|"
    r"%2e%2e%2f|/etc/passwd|;\s*(cat|ls|id|whoami)|\|\s*(id|whoami)|\$\(|`|\{\{.*?\}\}|"
    r"\$\{.*?\}|<%=|%27|%3cscript|\$ne\b|\$gt\b|\$where\b|169\.254\.169\.254|gopher://|"
    r"/\.git|/\.env|/wp-admin|/phpinfo|etc%2fpasswd|passwd|cmd=|exec)", re.I)

# each fold holds out one benign source B* and one attack spec A*. A* is a source id, or a
# "testbed" and tool pair for a tool-level holdout.
FOLDS = [
    ("zanbil", "srbh"),
    ("nasa", "testbed"),
    ("srbh", "csic"),
    ("clarknet", "srbh"),
    ("calgary", "testbed"),
    ("zanbil", ("testbed", "custom")),
]

_SIGCOLS = ["http_method", "url_path_raw", "url_query", "request_body"]


def _sig(df):
    return (df.http_method.astype(str) + "|" + df.url_path_raw.astype(str) + "|"
            + df.url_query.astype(str) + "|" + df.request_body.astype(str))


def _cap(df, n, seed=C.SEED):
    return df.sample(n=min(n, len(df)), random_state=seed) if len(df) > n else df


def reqs(df):
    return df[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records")


def _load_all():
    out = {}
    for sid, loader in SRC.REGISTRY.items():
        d = loader()
        if d is not None and len(d):
            out[sid] = d
    return out


def assemble(all_src, bstar, astar):
    astar_src = astar[0] if isinstance(astar, tuple) else astar
    held = {bstar, astar_src}

    train_ben, train_atk = [], []
    for sid, d in all_src.items():
        if sid in held:
            continue
        b = d[d.label == "benign"]
        a = d[d.label == "attack"]
        if len(b):
            train_ben.append(_cap(b, C.PER_SOURCE_BENIGN_CAP))
        if len(a):
            train_atk.append(_cap(a, C.PER_SOURCE_ATTACK_CAP))

    # tool-level attack holdout: keep testbed's non-A* attacks + benign in train
    if isinstance(astar, tuple):
        sid, tool = astar
        d = all_src[sid]
        train_ben.append(_cap(d[d.label == "benign"], C.PER_SOURCE_BENIGN_CAP))
        keep = d[(d.label == "attack") & (d.get("attack_tool") != tool)]
        if len(keep):
            train_atk.append(_cap(keep, C.PER_SOURCE_ATTACK_CAP))

    tb = pd.concat(train_ben, ignore_index=True)
    ta = pd.concat(train_atk, ignore_index=True)

    # calib holds LOSO_CALIB_FRAC of the real-benign train rows and the same fraction of
    # attacks, so Platt scaling and the threshold see both classes. Calib benign is real and
    # never comes from B*.
    real_ben = tb[tb.nature == "real"]
    calib_ben = real_ben.sample(frac=C.LOSO_CALIB_FRAC, random_state=C.SEED)
    calib_atk = ta.sample(frac=C.LOSO_CALIB_FRAC, random_state=C.SEED)
    fit_ben = tb.drop(index=calib_ben.index)
    fit_atk = ta.drop(index=calib_atk.index)

    fit = pd.concat([fit_ben.assign(y=0), fit_atk.assign(y=1)], ignore_index=True)
    calib = pd.concat([calib_ben.assign(y=0), calib_atk.assign(y=1)], ignore_index=True)

    # held-out eval sets
    bstar_df = all_src[bstar]
    Bstar = _cap(bstar_df[bstar_df.label == "benign"], C.LOSO_BSTAR_EVAL)
    if isinstance(astar, tuple):
        sid, tool = astar
        d = all_src[sid]
        Astar = _cap(d[(d.label == "attack") & (d.get("attack_tool") == tool)], C.LOSO_ASTAR_EVAL)
    else:
        d = all_src[astar]
        Astar = _cap(d[d.label == "attack"], C.LOSO_ASTAR_EVAL)

    # conservative: exclude from B* and A* any signature present in train, so both are novel
    train_sig = set(_sig(pd.concat([fit, calib], ignore_index=True)))
    nB0, nA0 = len(Bstar), len(Astar)
    Bstar = Bstar.loc[~_sig(Bstar).isin(train_sig)]
    Astar = Astar.loc[~_sig(Astar).isin(train_sig)]

    # source-level disjointness assertion
    assert not (set(_sig(Bstar)) & train_sig), "B* overlaps train after exclusion"
    assert not (set(_sig(Astar)) & train_sig), "A* overlaps train after exclusion"

    meta = {"fit_benign": int(len(fit_ben)), "fit_attack": int(len(fit_atk)),
            "calib_benign": int(len(calib_ben)), "calib_attack": int(len(calib_atk)),
            "train_benign_sources": sorted(set(tb.source_id)),
            "train_attack_sources": sorted(set(ta.source_id)),
            "Bstar_n": int(len(Bstar)), "Bstar_excluded_shared": int(nB0 - len(Bstar)),
            "Astar_n": int(len(Astar)), "Astar_excluded_shared": int(nA0 - len(Astar))}
    return fit, calib, Bstar, Astar, meta


def chan_thr(scores_calib, y_calib, fpr):
    b = np.asarray(scores_calib)[np.asarray(y_calib) == 0]
    return float(np.quantile(b, 1 - fpr, method="higher")) if len(b) else 0.5


def batched(det, df, which, batch=40_000):
    out = []
    for i in range(0, len(df), batch):
        r = reqs(df.iloc[i:i + batch])
        pt = det.p_text(r); pn = det.p_numeric(r)
        out.append(pt if which == "text" else det.p_content(r, pt, pn))
    return np.concatenate(out) if out else np.array([])


def run_fold(all_src, bstar, astar):
    fit, calib, Bstar, Astar, meta = assemble(all_src, bstar, astar)
    det = ContentDetector().fit(reqs(fit), fit.y.values, reqs(calib), calib.y.values)

    rc = reqs(calib)
    ptc = det.p_text(rc)
    res = {"Bstar": bstar, "Astar": (f"{astar[0]}:{astar[1]}" if isinstance(astar, tuple) else astar),
           "meta": meta}
    scored = {"B": {"text": batched(det, Bstar, "text"), "content": batched(det, Bstar, "content")},
              "A": {"text": batched(det, Astar, "text"), "content": batched(det, Astar, "content")}}
    for a in C.FPR_BUDGETS:
        th = {"text": chan_thr(ptc, calib.y, a), "content": det.threshold(a)}
        res[f"fpr_{a}"] = {
            "text_only": {"heldout_benign_alert": float((scored["B"]["text"] >= th["text"]).mean()),
                          "heldout_attack_recall": float((scored["A"]["text"] >= th["text"]).mean())},
            "fused": {"heldout_benign_alert": float((scored["B"]["content"] >= th["content"]).mean()),
                      "heldout_attack_recall": float((scored["A"]["content"] >= th["content"]).mean())},
        }
    # WAMM manual check: top-100 B* alerts by fused score
    thr = det.threshold(C.PRIMARY_FPR)
    B = Bstar.reset_index(drop=True).copy(); B["_pc"] = scored["B"]["content"]
    top = B[B["_pc"] >= thr].sort_values("_pc", ascending=False).head(C.MANUAL_INSPECT_N)
    marked = top.apply(lambda r: bool(_AUDIT.search(f"{r.url_path_raw} {r.url_query} {r.request_body}")), axis=1)
    res["wamm_manual"] = {"n_top_alerts": int(len(top)),
                          "regex_marked_plausible_attack": int(marked.sum()),
                          "true_fp_estimate": int((~marked).sum())}
    return res


def main():
    os.makedirs(C.MODEL_DIR_LOSO, exist_ok=True)
    all_src = _load_all()
    registry = {sid: {"nature": d.nature.iloc[0], "has_body": bool(d.has_body.iloc[0]),
                      "benign": int((d.label == "benign").sum()), "attack": int((d.label == "attack").sum())}
                for sid, d in all_src.items()}
    folds = []
    for bstar, astar in FOLDS:
        astar_src = astar[0] if isinstance(astar, tuple) else astar
        if bstar not in all_src or astar_src not in all_src:
            continue
        print(f"[fold] B*={bstar} A*={astar} ...", flush=True)
        folds.append(run_fold(all_src, bstar, astar))
    out = {"registry": registry, "folds": folds, "config": {
        "seed": C.SEED, "per_source_benign_cap": C.PER_SOURCE_BENIGN_CAP,
        "per_source_attack_cap": C.PER_SOURCE_ATTACK_CAP, "fpr_budgets": list(C.FPR_BUDGETS)}}
    with open(os.path.join(C.MODEL_DIR_LOSO, "loso_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    _print(out)
    return out


def _print(out):
    print("\n===== SOURCE REGISTRY =====")
    for sid, m in out["registry"].items():
        print(f"  {sid:10s} {m['nature']:16s} body={str(m['has_body'])[0]} benign={m['benign']:7d} attack={m['attack']:7d}")
    print("\n===== LEAVE-ONE-SOURCE-OUT FOLDS =====")
    for f in out["folds"]:
        print(f"\n  B*={f['Bstar']}  A*={f['Astar']}  "
              f"(train benign src={f['meta']['train_benign_sources']}, attack src={f['meta']['train_attack_sources']})")
        print(f"    B* n={f['meta']['Bstar_n']} (excl shared {f['meta']['Bstar_excluded_shared']}), "
              f"A* n={f['meta']['Astar_n']}")
        for a in C.FPR_BUDGETS:
            b = f[f"fpr_{a}"]
            print(f"    @{a}: text-only  alert={b['text_only']['heldout_benign_alert']*100:6.2f}%  recall={b['text_only']['heldout_attack_recall']*100:6.2f}%   "
                  f"| fused alert={b['fused']['heldout_benign_alert']*100:6.2f}%  recall={b['fused']['heldout_attack_recall']*100:6.2f}%")
        w = f["wamm_manual"]
        print(f"    WAMM top-{w['n_top_alerts']}: regex-marked plausible-attack={w['regex_marked_plausible_attack']}, true-FP est={w['true_fp_estimate']}")


if __name__ == "__main__":
    main()
