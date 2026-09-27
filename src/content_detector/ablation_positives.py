"""Retrains the in-domain zanbil model with the split, seed, hyperparameters and operating point
of in_domain_train, once with every testbed attack family among the positives and once with the
content families only. Writes models/in_domain/ablation_positives_results.json, which backs
Section 5.9.1.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config as C
from .in_domain_train import TextOnly, _sig, build_indomain_data
from .multidomain import load_modsec

MDIR = "models/in_domain"
TESTBED = "data/corpus/labelled/testbed.parquet"


def main():
    os.makedirs(MDIR, exist_ok=True)
    d = build_indomain_data()
    train_ben, calib_ben, test_ben = d["train_ben"], d["calib_ben"], d["test_ben"]
    fit_atk, srbh_te, tb_te = d["fit_atk"], d["srbh_te"], d["tb_te"]
    modsec = load_modsec()
    content = set(C.CONTENT_FAMILIES)

    # the loader does not carry attack_family; recover it from the labeled corpus by request signature
    tb_full = pd.read_parquet(TESTBED)
    tb_full["request_body"] = tb_full["request_body"].fillna("").astype(str)
    fam_map = dict(zip(_sig(tb_full), tb_full["attack_family"]))
    is_testbed = fit_atk["source_id"].eq("testbed")
    fam = _sig(fit_atk).map(fam_map).where(is_testbed)
    tb_te = tb_te.copy(); tb_te["attack_family"] = _sig(tb_te).map(fam_map)
    keep_B = ~is_testbed | fam.isin(content)
    variants = {"A_all_families": fit_atk, "B_content_only": fit_atk[keep_B]}
    out = {"n_fit_attack": {k: int(len(v)) for k, v in variants.items()},
           "n_testbed_removed_in_B": int((is_testbed & ~fam.isin(content)).sum()),
           "removed_families": sorted(set(fam[is_testbed & ~fam.isin(content)].dropna()))}

    for name, atk in variants.items():
        fit_df = pd.concat([train_ben.assign(y=0), atk.assign(y=1)], ignore_index=True)
        m = TextOnly().fit(fit_df, fit_df.y.values)
        p_cal = m.p(calib_ben)
        thr = {a: float(np.quantile(p_cal, 1 - a, method="higher")) for a in C.FPR_BUDGETS}
        p_late, p_srbh, p_tool = m.p(test_ben), m.p(srbh_te), m.p(tb_te)
        p_ms = m.p(modsec) if modsec is not None else None
        res = {}
        for a in C.FPR_BUDGETS:
            res[f"fpr_{a}"] = {"threshold": thr[a],
                               "late_zanbil_alert_rate": float((p_late >= thr[a]).mean()),
                               "srbh_attack_recall": float((p_srbh >= thr[a]).mean()),
                               "testbed_custom_tool_recall": float((p_tool >= thr[a]).mean()),
                               "modsec_recall": (float((p_ms >= thr[a]).mean()) if p_ms is not None else None)}
        # custom-tool recall split into content families and the rest, at the primary budget
        if "attack_family" in tb_te:
            grp = tb_te["attack_family"].isin(content).values
            res["custom_tool_recall_content_families@1pct"] = (
                float((p_tool[grp] >= thr[C.PRIMARY_FPR]).mean()) if grp.any() else None)
            res["custom_tool_recall_other_families@1pct"] = (
                float((p_tool[~grp] >= thr[C.PRIMARY_FPR]).mean()) if (~grp).any() else None)
            res["n_custom_tool_content"] = int(grp.sum()); res["n_custom_tool_other"] = int((~grp).sum())
        out[name] = res
        print(name, json.dumps(res, indent=1))

    with open(os.path.join(MDIR, "ablation_positives_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("n_fit_attack:", out["n_fit_attack"], "removed:", out["n_testbed_removed_in_B"], out["removed_families"])
    return out


if __name__ == "__main__":
    main()
