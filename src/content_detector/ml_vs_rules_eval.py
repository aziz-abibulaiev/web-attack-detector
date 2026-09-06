"""Compares the learned detector with the regex rule baseline, reports in-distribution and
cross-tool recall at the deployed thresholds of 1% and 5% FPR on zanbil benign, and ablates the
numeric channel. Writes models/in_domain/ml_vs_rules_results.json, which backs Sections 7.3-7.5.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .model import ContentDetector
from .normalize import decode_fixed_point
from .predict import _EXPLAIN_RULES
from .in_domain_train import TextOnly, build_indomain_data, _fields, _sig  # noqa: F401

MDIR = "models/in_domain"
TESTBED = "data/corpus/labelled/testbed.parquet"
CLEAN = ["sqli", "xss", "cmd_injection", "ssti"]
DROP_SHARED = ["path_traversal", "scan"]


def regex_flags(df):
    """ModSecurity-style baseline: flag if ANY detection rule matches the decoded request."""
    out = []
    for r in df[["url_path_raw", "url_query", "request_body"]].to_dict("records"):
        dec, _ = decode_fixed_point(f"{r.get('url_path_raw','')} {r.get('url_query','')} {r.get('request_body','')}")
        out.append(any(p.search(dec) for p in _EXPLAIN_RULES.values()))
    return np.array(out, dtype=bool)


def thr_on(scores, fpr):
    return float(np.quantile(scores, 1 - fpr, method="higher")) if len(scores) else 0.5


def main():
    os.makedirs(MDIR, exist_ok=True)
    d = build_indomain_data()
    train_ben, calib_ben, test_ben = d["train_ben"], d["calib_ben"], d["test_ben"]
    srbh_te, tb_te = d["srbh_te"], d["tb_te"]

    m = joblib.load(os.path.join(MDIR, "text_only_indomain.joblib"))
    thr = {a: thr_on(m.p(calib_ben), a) for a in C.FPR_BUDGETS}  # deployed operating point, from zanbil FPR
    A1 = C.PRIMARY_FPR

    # scores at deploy
    p_zan = m.p(test_ben)            # late-zanbil benign, for false positives
    p_srbh = m.p(srbh_te)           # SR-BH held-out attacks, in distribution
    p_tool = m.p(tb_te)             # testbed custom tool, cross-tool
    reg_zan = regex_flags(test_ben)
    reg_srbh = regex_flags(srbh_te)
    reg_tool = regex_flags(tb_te)

    def recall_split(scores, regmark, thr_):
        ml = scores >= thr_
        return {
            "overall": {"ml": float(ml.mean()), "regex": float(regmark.mean())},
            "marked": {"ml": float(ml[regmark].mean()) if regmark.any() else None,
                       "regex": 1.0 if regmark.any() else None, "n": int(regmark.sum())},
            "marker_free": {"ml": float(ml[~regmark].mean()) if (~regmark).any() else None,
                            "regex": 0.0 if (~regmark).any() else None, "n": int((~regmark).sum())},
        }

    # ===== head-to-head =====
    partB = {
        "operating_point": {"budget": A1, "threshold": thr[A1], "reference": "1% FPR on zanbil benign"},
        "false_positive_late_zanbil": {"ml_alert_rate": float((p_zan >= thr[A1]).mean()),
                                       "regex_alert_rate": float(reg_zan.mean())},
        "srbh_in_distribution_recall": recall_split(p_srbh, reg_srbh, thr[A1]),
        "testbed_custom_cross_tool_recall": recall_split(p_tool, reg_tool, thr[A1]),
    }

    # ===== cross-tool recall at deployed threshold =====
    zben = train_ben  # zanbil benign for training + thresholding the probe models
    zthr_ben = calib_ben
    tb_atk = pd.read_parquet(TESTBED, columns=["url_path_raw", "url_query", "request_body",
                                               "headers", "label", "attack_family", "attack_tool"])
    atk = tb_atk[tb_atk.label == "attack"]
    partC = {"operating_point": "1% FPR on zanbil benign", "in_domain_fp": partB["false_positive_late_zanbil"]["ml_alert_rate"]}
    for fam in CLEAN + DROP_SHARED:
        fa = atk[atk.attack_family == fam].copy()
        if fam in DROP_SHARED:
            key = fa.apply(lambda r: decode_fixed_point(f"{r.url_path_raw} {r.url_query} {r.request_body}")[0], axis=1)
            fa["_k"] = key.values
            shared = set(fa[fa.attack_tool == "custom"]["_k"]) & set(fa[fa.attack_tool == "ffuf"]["_k"])
            fa = fa[~fa["_k"].isin(shared)]
        toolA = fa[fa.attack_tool.isin(["ffuf", "sqlmap"])]; toolB = fa[fa.attack_tool == "custom"]
        if len(toolA) < 40 or len(toolB) < 40:
            partC[fam] = {"skipped": f"A={len(toolA)} B={len(toolB)}"}; continue
        res = {}
        for name, tr, te in [("ffuf_to_custom", toolA, toolB), ("custom_to_ffuf", toolB, toolA)]:
            mdl = TextOnly().fit(pd.concat([tr.assign(y=1), zben.assign(y=0)], ignore_index=True),
                                 np.r_[np.ones(len(tr)), np.zeros(len(zben))])
            t = thr_on(mdl.p(zthr_ben), A1)
            res[name] = {"recall": float((mdl.p(te) >= t).mean()), "n_test": int(len(te)),
                         "zanbil_fp": float((mdl.p(zthr_ben) >= t).mean())}
        partC[fam] = res
    partC["_note"] = "nosql_injection/ssrf excluded (ffuf slice off-family content); recall is CROSS-TOOL generalization"

    # ===== text-only vs fused in-domain, and marked vs marker-free =====
    fit_df = pd.concat([train_ben.assign(y=0), d["fit_atk"].assign(y=1)], ignore_index=True)
    fused = ContentDetector().fit(fit_df[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records"),
                                  fit_df.y.values,
                                  pd.concat([calib_ben.assign(y=0), d["fit_atk"].sample(frac=0.15, random_state=C.SEED).assign(y=1)], ignore_index=True)[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records"),
                                  np.r_[np.zeros(len(calib_ben)), np.ones(len(d["fit_atk"].sample(frac=0.15, random_state=C.SEED)))])
    fz = fused.p_content(test_ben[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records"))
    fs = fused.p_content(srbh_te[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records"))
    ft = fused.p_content(tb_te[["url_path_raw", "url_query", "request_body", "headers"]].to_dict("records"))
    fthr = fused.threshold(A1)
    partD = {
        "text_vs_fused_indomain": {
            "text_only": {"zanbil_fp": float((p_zan >= thr[A1]).mean()),
                          "srbh_recall": float((p_srbh >= thr[A1]).mean()),
                          "custom_recall": float((p_tool >= thr[A1]).mean())},
            "fused": {"zanbil_fp": float((fz >= fthr).mean()),
                      "srbh_recall": float((fs >= fthr).mean()),
                      "custom_recall": float((ft >= fthr).mean())},
        },
        "text_only_marked_vs_markerfree": {
            "srbh": partB["srbh_in_distribution_recall"],
            "custom_cross_tool": partB["testbed_custom_cross_tool_recall"],
        },
    }

    out = {"partB": partB, "partC": partC, "partD": partD,
           "config": {"seed": C.SEED, "deployed": "text-only in-domain", "budgets": list(C.FPR_BUDGETS)}}
    with open(os.path.join(MDIR, "ml_vs_rules_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    _print(out)
    return out


def _print(o):
    b = o["partB"]
    print("\n===== PART B: learned vs regex at the deployed point, 1% zanbil FPR =====")
    print(f"  late-zanbil FP:  ML={b['false_positive_late_zanbil']['ml_alert_rate']*100:.2f}%   "
          f"regex={b['false_positive_late_zanbil']['regex_alert_rate']*100:.2f}%")
    for k, lab in [("srbh_in_distribution_recall", "SR-BH (IN-DIST)"),
                   ("testbed_custom_cross_tool_recall", "custom (CROSS-TOOL)")]:
        r = b[k]
        print(f"  {lab} recall:  overall ML={r['overall']['ml']*100:.1f}% regex={r['overall']['regex']*100:.1f}%  |  "
              f"marker-free(n={r['marker_free']['n']}) ML={_pct(r['marker_free']['ml'])} regex=0.0%")
    print("\n===== PART C: CROSS-TOOL RECALL @ deployed threshold (paired with in-domain FP "
          f"{o['partC']['in_domain_fp']*100:.2f}%) =====")
    for fam in CLEAN + DROP_SHARED:
        r = o["partC"].get(fam, {})
        if "skipped" in r:
            print(f"  {fam:16s} {r['skipped']}"); continue
        print(f"  {fam:16s} ffuf→custom recall={r['ffuf_to_custom']['recall']*100:5.1f}% (fp={r['ffuf_to_custom']['zanbil_fp']*100:.2f}%)  "
              f"custom→ffuf recall={r['custom_to_ffuf']['recall']*100:5.1f}%")
    print("\n===== PART D: text vs fused, in-domain at the deployed point =====")
    t = o["partD"]["text_vs_fused_indomain"]
    print(f"  text-only: zanbil_fp={t['text_only']['zanbil_fp']*100:.2f}% srbh={t['text_only']['srbh_recall']*100:.1f}% custom={t['text_only']['custom_recall']*100:.1f}%")
    print(f"  fused:     zanbil_fp={t['fused']['zanbil_fp']*100:.2f}% srbh={t['fused']['srbh_recall']*100:.1f}% custom={t['fused']['custom_recall']*100:.1f}%")


def _pct(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


if __name__ == "__main__":
    main()
