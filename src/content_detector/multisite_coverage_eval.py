"""Two diagnostics for the WebLog temporal split. The first reuses the saved temporal model and
attributes each alert in the late test window to a subdomain, reporting whether that subdomain
appears in the training slice at all. The second retrains with the split stratified by
subdomain, so time still separates train from test but every subdomain is represented. Writes
models/weblog_edgar/multisite_coverage_results.json and multisite_coverage_sites.csv
which back Sections 4.5 and 7.2.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import multidomain as MD
from .in_domain_train import TextOnly  # noqa: F401 — needed to unpickle and to retrain

MD.MDIR = "models/weblog_edgar"
MDIR = MD.MDIR
EXT = "data/corpus/external"
MIN_SITE_N = 50          # segments below this go into one bucket
SITE_RE = r"^/([^/?]*)"


def site_key(paths: pd.Series) -> pd.Series:
    s = paths.astype(str).str.extract(SITE_RE, expand=False).fillna("").str.lower()
    return s.replace("", "(root)")


def load_benign():
    wl = pd.read_parquet(f"{EXT}/weblog2025.parquet")
    ben = wl[wl.label == "benign"].copy()
    ben["ts"] = pd.to_datetime(ben.timestamp, errors="coerce")
    ben = ben[ben.ts.notna() & (ben.ts.dt.year > 1970)]
    ben["site"] = site_key(ben.url_path_raw)
    # collapse rare first segments, mostly page slugs of root sites, into one bucket
    vc = ben.site.value_counts()
    small = set(vc[vc < MIN_SITE_N].index)
    ben.loc[ben.site.isin(small), "site"] = "(small)"
    return ben


# --------------------------------------------------------------------- A: attribution
def part_a_attribution(ben):
    train_b, calib_b, test_b, method = MD.split_benign(ben, "temporal")
    boundary = pd.Timestamp(method.split("@", 1)[1])
    m = joblib.load(os.path.join(MDIR, "text_only_weblog.joblib"))
    thr1 = float(np.quantile(MD._batched_p(m, calib_b), 1 - 0.01, method="higher"))
    p = MD._batched_p(m, test_b)
    T = test_b.reset_index(drop=True).copy()
    T["alert"] = p >= thr1

    overall = float(T.alert.mean())
    print(f"[A] temporal split reproduced: {method}")
    print(f"[A] overall late-benign alert @1% = {overall*100:.3f}%  reference 20.024%")

    train_sites = set(train_b.site)
    first_seen = ben.groupby("site").ts.min()
    g = T.groupby("site").agg(n_test=("alert", "size"), alerts=("alert", "sum"))
    g["alert_rate"] = g.alerts / g.n_test
    g["in_train_slice"] = g.index.isin(train_sites)
    g["first_seen"] = first_seen.reindex(g.index)
    g["appeared_after_split"] = g.first_seen >= boundary
    g = g.sort_values("alerts", ascending=False)
    g.to_csv(os.path.join(MDIR, "multisite_coverage_sites.csv"))

    cov = T[T.site.isin(train_sites)]
    unc = T[~T.site.isin(train_sites)]
    total_fp = int(T.alert.sum())
    res = {
        "split_method": method, "threshold_1pct": thr1,
        "overall_alert_rate": overall,
        "n_sites_in_test": int(T.site.nunique()),
        "n_sites_covered_by_train_slice": int(T.site.isin(train_sites).sum() and cov.site.nunique()),
        "covered_sites": {"n_test": int(len(cov)), "alert_rate": float(cov.alert.mean()) if len(cov) else None},
        "uncovered_sites": {"n_test": int(len(unc)), "alert_rate": float(unc.alert.mean()) if len(unc) else None,
                            "n_sites": int(unc.site.nunique())},
        "share_of_all_fp_from_uncovered_sites": (float(unc.alert.sum() / total_fp) if total_fp else None),
        "top15_sites_by_fp": [
            {"site": s, "n_test": int(r.n_test), "alerts": int(r.alerts),
             "alert_rate": float(r.alert_rate), "in_train_slice": bool(r.in_train_slice),
             "appeared_after_split": (None if pd.isna(r.appeared_after_split) else bool(r.appeared_after_split))}
            for s, r in g.head(15).iterrows()],
    }
    print(f"[A] covered sites  : n={len(cov):6d}  alert-rate={cov.alert.mean()*100 if len(cov) else 0:.3f}%")
    print(f"[A] uncovered sites: n={len(unc):6d}  alert-rate={unc.alert.mean()*100 if len(unc) else 0:.3f}%"
          f"  ({unc.site.nunique()} sites)")
    if total_fp:
        print(f"[A] share of ALL false alerts coming from uncovered sites: {unc.alert.sum()/total_fp*100:.1f}%")
    return res


# ---------------------------------------------------- B: site-stratified temporal split
def _sig(df):
    return MD._sig(df)


def part_b_stratified(ben, atk, modsec):
    early_parts, late_parts = [], []
    for _, gdf in ben.groupby("site"):
        gdf = gdf.sort_values("ts")
        if len(gdf) < 10:
            early_parts.append(gdf)
            continue
        b = gdf.ts.quantile(MD.TEMPORAL_Q)
        early_parts.append(gdf[gdf.ts < b])
        late_parts.append(gdf[gdf.ts >= b])
    early = pd.concat(early_parts, ignore_index=True).sample(frac=1.0, random_state=C.SEED)
    late = pd.concat(late_parts, ignore_index=True)

    n_calib = min(25_000, max(1, int(len(early) * 0.15)))
    calib = early.iloc[:n_calib]
    train = early.iloc[n_calib:n_calib + 120_000]
    test = late.sample(n=min(100_000, len(late)), random_state=C.SEED)
    assert not (set(_sig(train)) & set(_sig(test))), "train/test overlap"
    assert not (set(_sig(train)) & set(_sig(calib))), "train/calib overlap"
    assert not (set(_sig(calib)) & set(_sig(test))), "calib/test overlap"
    print(f"[B] stratified: early={len(early)} late={len(late)} -> train={len(train)} "
          f"calib={len(calib)} test={len(test)}; every-site-in-train check: "
          f"{test.site.isin(set(train.site)).mean()*100:.2f}% of test rows from trained sites")

    fit = pd.concat([train.assign(y=0), atk["train"].assign(y=1)], ignore_index=True)
    m = TextOnly().fit(pd.DataFrame(MD._reqcols(fit)), fit.y.values)
    thr = {a: float(np.quantile(MD._batched_p(m, calib), 1 - a, method="higher")) for a in C.FPR_BUDGETS}
    p = MD._batched_p(m, test)
    p_srbh = MD._batched_p(m, atk["srbh_te"]); p_custom = MD._batched_p(m, atk["custom_te"])
    p_modsec = MD._batched_p(m, modsec) if modsec is not None else None

    res = {"benign_train": int(len(train)), "benign_calib": int(len(calib)), "benign_test": int(len(test)),
           "test_rows_from_trained_sites": float(test.site.isin(set(train.site)).mean())}
    for a in C.FPR_BUDGETS:
        res[f"fpr_{a}"] = {
            "late_benign_alert": float((p >= thr[a]).mean()),
            "srbh_recall": float((p_srbh >= thr[a]).mean()),
            "testbed_custom_cross_tool_recall": float((p_custom >= thr[a]).mean()),
            "modsec_recall": (float((p_modsec >= thr[a]).mean()) if p_modsec is not None else None)}
    b1 = res["fpr_0.01"]
    print(f"[B] STRATIFIED-TEMPORAL FP @1% = {b1['late_benign_alert']*100:.3f}%  "
          f"@5% = {res['fpr_0.05']['late_benign_alert']*100:.3f}%")
    print(f"[B] recall @1%: SR-BH={b1['srbh_recall']*100:.1f}%  cross-tool="
          f"{b1['testbed_custom_cross_tool_recall']*100:.1f}%  ModSec={(b1['modsec_recall'] or 0)*100:.1f}%")
    return res


def main():
    os.makedirs(MDIR, exist_ok=True)
    ben = load_benign()
    print(f"[multisite_coverage] WebLog benign={len(ben)}  sites(>= {MIN_SITE_N} req)={ben.site.nunique()}")
    out = {"config": {"seed": C.SEED, "min_site_n": MIN_SITE_N}}
    out["A_attribution"] = part_a_attribution(ben)
    atk = MD.build_attacks()
    modsec = MD.load_modsec()
    out["B_stratified_temporal"] = part_b_stratified(ben, atk, modsec)
    with open(os.path.join(MDIR, "multisite_coverage_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[multisite_coverage] -> {MDIR}/multisite_coverage_results.json")


if __name__ == "__main__":
    main()
