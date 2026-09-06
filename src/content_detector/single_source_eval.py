"""Trains with one real benign source and measures the alert rate on a different one, in both
directions: SR-BH benign in training and zanbil as the test source, then the reverse. The
threshold comes from the FPR budget on a calibration slice, and the run asserts that the test
source never appears in training. Writes models/single_source/single_source_results.json
which backs Section 7.1.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from . import data_real as D
from .model import ContentDetector

TESTBED = "data/corpus/labelled/testbed.parquet"

# inspection aid for manual review only, never a feature and never a training filter
_AUDIT = re.compile(
    r"(union\s+select|\bor\s+1=1\b|'\s*or\s*'1'='1|<script|onerror\s*=|javascript:|\.\./|"
    r"%2e%2e%2f|/etc/passwd|;\s*(cat|ls|id|whoami)|\|\s*(id|whoami)|\$\(|`|\{\{.*?\}\}|"
    r"\$\{.*?\}|<%=|%27|%3cscript|\$ne\b|\$gt\b|\$where\b|169\.254\.169\.254|gopher://|"
    r"/\.git|/\.env|/wp-admin|/phpinfo|etc%2fpasswd|passwd|cmd=|exec)", re.I)


def reqs(df):
    return df[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records")


def batched_scores(det, df, which="content", batch=40_000):
    out = []
    for i in range(0, len(df), batch):
        part = df.iloc[i:i + batch]
        r = reqs(part)
        pt = det.p_text(r); pn = det.p_numeric(r)
        out.append(pt if which == "text" else pn if which == "numeric" else det.p_content(r, pt, pn))
    return np.concatenate(out) if out else np.array([])


def chan_thr(scores_calib, y_calib, fpr):
    b = np.asarray(scores_calib)[np.asarray(y_calib) == 0]
    return float(np.quantile(b, 1 - fpr, method="higher")) if len(b) else 0.5


def _testbed_attack_split(atk):
    """Split testbed attacks by unique payload signature per family, 70/15/15 into fit, calib
    and test, so no payload leaks across slices. Every tool appears in every slice."""
    sig = (atk.attack_family.astype(str) + "|" + atk.url_path_raw.astype(str) + "|"
           + atk.url_query.astype(str) + "|" + atk.request_body.astype(str))
    atk = atk.copy(); atk["_sig"] = sig
    uniq = atk.drop_duplicates("_sig")

    def bucket(k):
        h = int(hashlib.sha256(f"{C.SEED}:{k}".encode()).hexdigest(), 16) % 10_000 / 10_000.0
        return "fit" if h < 0.70 else ("calib" if h < 0.85 else "test")
    b = {k: bucket(k) for k in uniq["_sig"]}
    atk["_bk"] = atk["_sig"].map(b)
    return atk[atk._bk == "fit"], atk[atk._bk == "calib"], atk[atk._bk == "test"]


def run_config(name, tb_train_ben, tb_fit_atk, tb_calib_atk, tb_test_atk,
               real_ben_train, real_ben_calib, fpr_test_df, fpr_test_name, srbh_atk):
    # ---- disjointness ----
    # Within a source, real-benign train and calib must not share a row.
    train_ben_keys = D.sig_keys(real_ben_train)
    calib_ben_keys = D.sig_keys(real_ben_calib)
    assert not (train_ben_keys & calib_ben_keys), f"{name}: real-benign train/calib share rows"
    # The FPR-test source is a different site, capture and era from the training source.
    # Two real sources still share trivial requests such as GET / and /favicon.ico. Those are
    # not leakage, but they are excluded from the FPR test so every row in it is novel to the
    # model and a row the model has seen can never lower the reported alert rate.
    fpr_keys = D.sig_keys(fpr_test_df)
    shared = train_ben_keys & fpr_keys
    n_before = len(fpr_test_df)
    if shared:
        fpr_test_df = fpr_test_df.loc[~D._sig(fpr_test_df).isin(shared)].copy()
    n_excluded = n_before - len(fpr_test_df)

    # ---- assemble train / calib ----
    train_ben = pd.concat([tb_train_ben, real_ben_train], ignore_index=True)
    train_df = pd.concat([train_ben.assign(y=0), tb_fit_atk.assign(y=1)], ignore_index=True)
    calib_df = pd.concat([real_ben_calib.assign(y=0), tb_calib_atk.assign(y=1)], ignore_index=True)

    det = ContentDetector().fit(reqs(train_df), train_df.y.values, reqs(calib_df), calib_df.y.values)
    os.makedirs(C.MODEL_DIR_SINGLE_SOURCE, exist_ok=True)
    joblib.dump(det, os.path.join(C.MODEL_DIR_SINGLE_SOURCE, f"content_detector_{name}.joblib"))

    # calib channel scores for per-channel thresholds
    rc = reqs(calib_df); ptc = det.p_text(rc); pnc = det.p_numeric(rc)

    # eval sets
    S = {"fpr_benign": fpr_test_df, "srbh_attack": srbh_atk, "testbed_test_attack": tb_test_atk}
    scored = {k: {ch: batched_scores(det, v, ch) for ch in ("text", "numeric", "content")}
              for k, v in S.items()}

    res = {"fpr_test_source": fpr_test_name, "n": {k: int(len(v)) for k, v in S.items()},
           "fpr_test_shared_excluded": int(n_excluded),
           "train_benign_n": int(len(train_ben)), "train_attack_n": int(len(tb_fit_atk))}
    for a in C.FPR_BUDGETS:
        th = {"text": chan_thr(ptc, calib_df.y, a), "numeric": chan_thr(pnc, calib_df.y, a),
              "content": det.threshold(a)}
        res[f"fpr_{a}"] = {
            "thresholds": th,
            "heldout_benign_alert_rate": {ch: float((scored["fpr_benign"][ch] >= th[ch]).mean()) for ch in th},
            "srbh_attack_recall": {ch: float((scored["srbh_attack"][ch] >= th[ch]).mean()) for ch in th},
            "testbed_test_recall": {ch: float((scored["testbed_test_attack"][ch] >= th[ch]).mean()) for ch in th},
        }
    # threshold-free
    yb = np.r_[np.zeros(len(fpr_test_df)), np.ones(len(srbh_atk))]
    sc = np.r_[scored["fpr_benign"]["content"], scored["srbh_attack"]["content"]]
    res["roc_auc_content_real"] = float(roc_auc_score(yb, sc))
    res["pr_auc_content_real"] = float(average_precision_score(yb, sc))

    # ---- manual-inspection sample: top-N high-confidence alerts on held-out benign ----
    thr = det.threshold(C.PRIMARY_FPR)
    fb = fpr_test_df.reset_index(drop=True).copy()
    fb["_pc"] = scored["fpr_benign"]["content"]
    alerts = fb[fb["_pc"] >= thr].sort_values("_pc", ascending=False).head(C.MANUAL_INSPECT_N)
    marker = alerts.apply(lambda r: bool(_AUDIT.search(
        f"{r.url_path_raw} {r.url_query} {r.request_body}")), axis=1)
    res["manual_inspection"] = {
        "n_inspected": int(len(alerts)),
        "regex_marked_plausible_attack": int(marker.sum()),
        "true_fp_estimate": int((~marker).sum()),
        "note": "regex used as inspection aid only; alert-rate is an upper bound (WAMM: real "
                "benign carries ~9-10% hidden attacks)",
    }
    alerts.assign(regex_marked=marker.values)[
        ["http_method", "url_path_raw", "url_query", "_pc", "regex_marked"]
    ].to_csv(os.path.join(C.MODEL_DIR_SINGLE_SOURCE, f"manual_inspect_{name}.csv"), index=False)
    return res


def main():
    os.makedirs(C.MODEL_DIR_SINGLE_SOURCE, exist_ok=True)
    tb = pd.read_parquet(TESTBED)
    tb_ben = tb[tb.label == "benign"]
    tb_atk = tb[tb.label == "attack"]
    fit_atk, calib_atk, test_atk = _testbed_attack_split(tb_atk)

    srbh_atk = D.srbh_attacks(C.SRBH_ATTACK_EVAL_SAMPLE)

    # A1: SR-BH benign train/calib ; zanbil held-out FPR test
    sb_tr, sb_ca = D.srbh_benign_split()
    zan_fpr = D.zanbil_dedup(n_unique=C.ZANBIL_FPR_TEST_SAMPLE)
    a1 = run_config("A1", tb_ben, fit_atk, calib_atk, test_atk, sb_tr, sb_ca,
                    zan_fpr, "zanbil", srbh_atk)

    # A2: zanbil benign train/calib ; SR-BH benign held-out FPR test
    zan_tr, zan_ca, _ = D.zanbil_split()
    sb_all = D.srbh_benign_all()
    a2 = run_config("A2", tb_ben, fit_atk, calib_atk, test_atk, zan_tr, zan_ca,
                    sb_all, "srbh_benign", srbh_atk)

    out = {"A1": a1, "A2": a2, "config": {
        "seed": C.SEED, "real_benign_train_frac": C.REAL_BENIGN_TRAIN_FRAC,
        "zanbil_train_sample": C.ZANBIL_TRAIN_SAMPLE, "srbh_attack_eval_sample": C.SRBH_ATTACK_EVAL_SAMPLE,
        "wamm_hidden_attack_rate": C.WAMM_HIDDEN_ATTACK_RATE}}
    with open(os.path.join(C.MODEL_DIR_SINGLE_SOURCE, "single_source_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    _print(out)
    return out


def _print(out):
    for cfg in ("A1", "A2"):
        r = out[cfg]
        print(f"\n===== {cfg}: train benign source -> FPR test on {r['fpr_test_source']} "
              f"(n_benign={r['n']['fpr_benign']}, n_srbh_atk={r['n']['srbh_attack']}) =====")
        print(f"  ROC content-real={r['roc_auc_content_real']:.4f} PR={r['pr_auc_content_real']:.4f}")
        for a in C.FPR_BUDGETS:
            b = r[f"fpr_{a}"]
            print(f"  budget {a}:")
            for ch in ("text", "numeric", "content"):
                print(f"    {ch:8s} heldout_benign_alert={b['heldout_benign_alert_rate'][ch]*100:6.2f}%  "
                      f"srbh_recall={b['srbh_attack_recall'][ch]*100:6.2f}%  "
                      f"testbed_recall={b['testbed_test_recall'][ch]*100:6.2f}%")
        m = r["manual_inspection"]
        print(f"  manual-inspect top-{m['n_inspected']} alerts: regex-marked plausible-attack="
              f"{m['regex_marked_plausible_attack']}, true-FP est={m['true_fp_estimate']}")


if __name__ == "__main__":
    main()
