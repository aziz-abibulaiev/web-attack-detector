"""Trains the content detector on the testbed corpus and evaluates it there: binary core,
cross-tool holdout, marked versus marker-free recall, and the family head. Writes
models/lab_testbed/eval_results.json, content_detector.joblib and family_head.joblib
which back Sections 7.5 and 7.6.
"""

from __future__ import annotations

import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config as C
from .model import ContentDetector
from .split import assign_splits, split_frame

TESTBED = "data/corpus/labelled/testbed.parquet"

# used for the marker check only: a measurement instrument, never a feature or a label.
_AUDIT_MARKERS = re.compile(
    r"(union\s+select|\bor\s+1=1\b|'\s*or\s*'1'='1|<script|onerror\s*=|javascript:|"
    r"\.\./|%2e%2e%2f|/etc/passwd|;\s*(cat|ls|id|whoami)|\|\s*(id|whoami)|\$\(|`|"
    r"\{\{.*?\}\}|\$\{.*?\}|<%=|%27|%3cscript|\$ne\b|\$gt\b|\$where\b|169\.254\.169\.254|gopher://)",
    re.IGNORECASE,
)


def reqs_of(df) -> list[dict]:
    return df[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records")


def metrics(y, scores, thr) -> dict:
    y = np.asarray(y, int)
    pred = (np.asarray(scores) >= thr).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    out = {
        "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, scores)) if len(set(y)) > 1 else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "threshold": float(thr),
    }
    return out


def chan_threshold(scores_calib, y_calib, fpr):
    b = np.asarray(scores_calib)[np.asarray(y_calib) == 0]
    return float(np.quantile(b, 1 - fpr, method="higher")) if len(b) else 0.5


def main():
    os.makedirs(C.MODEL_DIR_LAB_TESTBED, exist_ok=True)
    df = pd.read_parquet(TESTBED)
    df["y"] = (df.label == "attack").astype(int)
    df, split_stats = split_frame(df)
    tr = df[df._split == "train"]; ca = df[df._split == "calib"]; te = df[df._split == "test"]

    results = {"split_stats": split_stats, "config": {
        "tfidf": C.TFIDF, "logreg": C.LOGREG, "hgb": C.HGB, "fusion": C.FUSION,
        "fusion_weights": C.FUSION_WEIGHTS, "fpr_budgets": list(C.FPR_BUDGETS), "seed": C.SEED}}

    # ===== binary core + late fusion =====
    det = ContentDetector().fit(reqs_of(tr), tr.y.values, reqs_of(ca), ca.y.values)
    r_te = reqs_of(te)
    pt = det.p_text(r_te); pn = det.p_numeric(r_te); pc = det.p_content(r_te, pt, pn)
    # per-channel thresholds from calib
    r_ca = reqs_of(ca)
    pt_ca = det.p_text(r_ca); pn_ca = det.p_numeric(r_ca)
    partA = {}
    for a in C.FPR_BUDGETS:
        partA[f"fpr_{a}"] = {
            "p_text": metrics(te.y, pt, chan_threshold(pt_ca, ca.y, a)),
            "p_numeric": metrics(te.y, pn, chan_threshold(pn_ca, ca.y, a)),
            "p_content": metrics(te.y, pc, det.threshold(a)),
        }
    results["partA"] = partA
    joblib.dump(det, os.path.join(C.MODEL_DIR_LAB_TESTBED, "content_detector.joblib"))

    # ===== cross-tool holdout on the clean families sqli, xss, cmd_injection, ssti =====
    partB = _cross_tool(df)
    results["partB"] = partB

    # ===== marked vs marker-free recall per channel, on the test attacks =====
    atk = te[te.y == 1].copy()
    marked_mask = atk.apply(lambda r: bool(_AUDIT_MARKERS.search(
        f"{r.url_path_raw} {r.url_query} {r.request_body}")), axis=1).values
    r_atk = reqs_of(atk)
    pt_a = det.p_text(r_atk); pn_a = det.p_numeric(r_atk); pc_a = det.p_content(r_atk, pt_a, pn_a)
    th = {c: det.threshold(C.PRIMARY_FPR) for c in ["p_content"]}
    th["p_text"] = chan_threshold(pt_ca, ca.y, C.PRIMARY_FPR)
    th["p_numeric"] = chan_threshold(pn_ca, ca.y, C.PRIMARY_FPR)
    def rec(scores, t, mask):
        s = np.asarray(scores)[mask]
        return float((s >= t).mean()) if len(s) else None
    results["partC"] = {
        "primary_fpr": C.PRIMARY_FPR,
        "n_marked": int(marked_mask.sum()), "n_marker_free": int((~marked_mask).sum()),
        "marked": {"p_text": rec(pt_a, th["p_text"], marked_mask),
                   "p_numeric": rec(pn_a, th["p_numeric"], marked_mask),
                   "p_content": rec(pc_a, th["p_content"], marked_mask)},
        "marker_free": {"p_text": rec(pt_a, th["p_text"], ~marked_mask),
                        "p_numeric": rec(pn_a, th["p_numeric"], ~marked_mask),
                        "p_content": rec(pc_a, th["p_content"], ~marked_mask)},
    }

    # ===== family head, testbed only =====
    results["partD"] = _family_head(tr, te)

    with open(os.path.join(C.MODEL_DIR_LAB_TESTBED, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    _print_report(results)
    return results


def _build_binary(train_df, calib_df):
    return ContentDetector().fit(reqs_of(train_df), train_df.y.values,
                                 reqs_of(calib_df), calib_df.y.values)


def _cross_tool(df):
    """Per-family cross-tool holdout in both directions on the clean families. path_traversal
    and scan run after their shared payloads are dropped; nosql_injection and ssrf are binary
    only, with no per-family holdout."""
    benign = df[df.label == "benign"]
    b_assign = assign_splits(benign)  # reuse hashing for benign train/calib/test
    benign = benign.copy(); benign["_b"] = benign.session_id.map(b_assign)
    b_tr = benign[benign._b == "train"]; b_ca = benign[benign._b == "calib"]; b_te = benign[benign._b == "test"]

    atk = df[df.label == "attack"].copy()
    out = {}
    fams = C.CLEAN_XTOOL_FAMILIES + C.DROP_SHARED_BEFORE_XTOOL
    for fam in fams:
        fa = atk[atk.attack_family == fam].copy()
        if fam in C.DROP_SHARED_BEFORE_XTOOL:
            # drop payloads shared between custom and ffuf, compared after decoding
            from .normalize import decode_fixed_point
            key = fa.apply(lambda r: decode_fixed_point(f"{r.url_path_raw} {r.url_query} {r.request_body}")[0], axis=1)
            fa["_key"] = key.values
            cset = set(fa[fa.attack_tool == "custom"]["_key"])
            fset = set(fa[fa.attack_tool == "ffuf"]["_key"])
            shared = cset & fset
            fa = fa[~fa["_key"].isin(shared)]
            out.setdefault("_shared_dropped", {})[fam] = len(shared)
        toolA = fa[fa.attack_tool.isin(["ffuf", "sqlmap"])]
        toolB = fa[fa.attack_tool == "custom"]
        if len(toolA) < 40 or len(toolB) < 40:
            out[fam] = {"skipped": f"insufficient rows A={len(toolA)} B={len(toolB)}"}
            continue

        def hold(fr):
            # deterministic 80/20 split of the training tool's attacks: fit vs fusion-calib
            fr = fr.sample(frac=1.0, random_state=C.SEED)
            k = max(20, int(len(fr) * 0.2))
            return fr.iloc[k:], fr.iloc[:k]  # fit, then calib

        res = {}
        # A->B : train ffuf/sqlmap, test custom; the test tool is untouched by fit and calib
        aFit, aCal = hold(toolA)
        det = _build_binary(pd.concat([aFit, b_tr]), pd.concat([aCal, b_ca]))
        thr = det.threshold(C.PRIMARY_FPR)
        res["A_ffuf_to_B_custom"] = {
            "recall_heldout": float((det.p_content(reqs_of(toolB)) >= thr).mean()),
            "benign_fpr": float((det.p_content(reqs_of(b_te)) >= thr).mean()),
            "n_train_attack": int(len(aFit)), "n_test_attack": int(len(toolB))}
        # B->A : train custom, test ffuf/sqlmap
        bFit, bCal = hold(toolB)
        det2 = _build_binary(pd.concat([bFit, b_tr]), pd.concat([bCal, b_ca]))
        thr2 = det2.threshold(C.PRIMARY_FPR)
        res["B_custom_to_A_ffuf"] = {
            "recall_heldout": float((det2.p_content(reqs_of(toolA)) >= thr2).mean()),
            "benign_fpr": float((det2.p_content(reqs_of(b_te)) >= thr2).mean()),
            "n_train_attack": int(len(bFit)), "n_test_attack": int(len(toolA))}
        out[fam] = res
    out["_excluded_binary_only"] = C.BINARY_ONLY_FFUF_OFFCONTENT
    return out


def _family_head(tr, te):
    # Family identification is a lexical task, sqli vs xss vs cmd tokens, so the
    # head uses char-TFIDF text, not shape features — shape cannot separate injection families.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from .model import ContentDetector
    tra = tr[tr.y == 1]; tea = te[te.y == 1]
    fa_tr, _ = ContentDetector._fields(reqs_of(tra))
    fa_te, _ = ContentDetector._fields(reqs_of(tea))
    vec = TfidfVectorizer(max_features=C.TFIDF_MAX_FEATURES_A, **C.TFIDF)
    Xtr = vec.fit_transform(fa_tr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0,
                             solver="lbfgs", random_state=C.SEED)
    clf.fit(Xtr, tra.attack_family.values)
    pred = clf.predict(vec.transform(fa_te))
    labels = sorted(set(tra.attack_family) | set(tea.attack_family))
    cm = confusion_matrix(tea.attack_family, pred, labels=labels)
    joblib.dump({"vectorizer": vec, "clf": clf, "labels": list(clf.classes_)},
                os.path.join(C.MODEL_DIR_LAB_TESTBED, "family_head.joblib"))
    per = {}
    for i, lab in enumerate(labels):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        per[lab] = {"support": int(cm[i].sum()),
                    "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                    "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0}
    return {"note": "TESTBED-MEASURED, not real-traffic-validated",
            "labels": labels, "confusion_matrix": cm.tolist(), "per_family": per}


def _numrow(r):
    from .features import numeric_features
    return numeric_features(url_path_raw=r.get("url_path_raw", ""), url_query=r.get("url_query", ""),
                            request_body=r.get("request_body", ""), headers=r.get("headers"))


def _print_report(res):
    print("\n===== SPLIT =====")
    for s, v in res["split_stats"]["splits"].items():
        print(f"  {s:6s} rows={v['rows']:6d} benign={v['benign']:6d} attack={v['attack']:5d} "
              f"sessions={v['sessions']} fams={v['attack_by_family']}")
    print("  disjoint:", res["split_stats"]["disjoint"])
    print("\n===== PART A: test =====")
    for a, chans in res["partA"].items():
        print(f"  {a}:")
        for ch, m in chans.items():
            print(f"    {ch:10s} ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} F1={m['f1']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} FPR={m['fpr']:.4f}")
    print("\n===== PART B: CROSS-TOOL HOLDOUT =====")
    for fam, r in res["partB"].items():
        if fam.startswith("_"):
            print(f"  {fam}: {r}"); continue
        if "skipped" in r:
            print(f"  {fam}: {r['skipped']}"); continue
        ab = r["A_ffuf_to_B_custom"]; ba = r["B_custom_to_A_ffuf"]
        print(f"  {fam:16s} ffuf→custom recall={ab['recall_heldout']:.3f} (fpr={ab['benign_fpr']:.3f}) | "
              f"custom→ffuf recall={ba['recall_heldout']:.3f} (fpr={ba['benign_fpr']:.3f})")
    print("\n===== PART C: marked vs marker-free recall =====")
    c = res["partC"]
    print(f"  marked(n={c['n_marked']}):      text={c['marked']['p_text']:.3f} "
          f"num={c['marked']['p_numeric']:.3f} fused={c['marked']['p_content']:.3f}")
    print(f"  marker_free(n={c['n_marker_free']}): text={c['marker_free']['p_text']:.3f} "
          f"num={c['marker_free']['p_numeric']:.3f} fused={c['marker_free']['p_content']:.3f}")
    print("\n===== PART D: family head, testbed only =====")
    for lab, m in res["partD"]["per_family"].items():
        print(f"  {lab:22s} support={m['support']:4d} P={m['precision']:.3f} R={m['recall']:.3f}")


if __name__ == "__main__":
    main()
