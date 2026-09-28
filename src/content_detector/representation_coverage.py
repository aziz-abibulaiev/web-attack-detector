"""Measures representation coverage, Definition 2 of the paper: the nearest-neighbor cosine
distance from each target benign request to the training benign set, in the detector's own
character n-gram representation. Writes models/coverage/representation_coverage_results.json,
which backs Section 7.8.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from . import config as C
from . import loso_eval as LO
from . import multidomain as MD
from .in_domain_train import TextOnly

MDIR = "models/coverage"
N_TARGET = 3000      # target benign rows sampled per pair
N_CALIB = 3000       # calibration benign rows used to set the radius
CHUNK = 250
ALPHAS = list(C.FPR_BUDGETS)
RHO = 0.01           # slack in the bound alert_rate <= eps + rho

# alert rates of the corresponding published runs, recorded next to each measurement
_REF_LOSO = {("zanbil", "srbh"): 0.9947, ("nasa", "testbed"): 0.00012, ("srbh", "csic"): 0.9805,
             ("clarknet", "srbh"): 0.00065, ("calgary", "testbed"): 0.0,
             ("zanbil", ("testbed", "custom")): 0.1962}


def _rows(df):
    return pd.DataFrame(MD._reqcols(df))


def _nn_dist(m, Xtr, df):
    """Cosine distance from each row of df to its nearest row of Xtr, both in the model's space."""
    Xq = normalize(m._X(_rows(df)))
    out = np.empty(Xq.shape[0])
    for i in range(0, Xq.shape[0], CHUNK):
        S = Xq[i:i + CHUNK] @ Xtr.T
        out[i:i + CHUNK] = 1.0 - np.asarray(S.max(axis=1).todense()).ravel()
    return out


def measure(name, train_b, calib_b, target_b, fit_atk, reference, t0):
    """Fit the text-only model on the published fit set, then compare, at each alpha, the share
    of target rows outside the radius r_alpha with the model's alert rate on the same rows. The
    radius is the 1 - alpha quantile of the calibration distances, the same rule that sets the
    threshold from the calibration scores."""
    print(f"[{name}] train={len(train_b)} calib={len(calib_b)} target={len(target_b)}", flush=True)
    fit = pd.concat([train_b.assign(y=0), fit_atk.assign(y=1)], ignore_index=True)
    m = TextOnly().fit(_rows(fit), fit.y.values)
    Xtr = normalize(m._X(_rows(train_b)))
    cal = calib_b.sample(n=min(N_CALIB, len(calib_b)), random_state=C.SEED)
    tgt = target_b.sample(n=min(N_TARGET, len(target_b)), random_state=C.SEED)
    d_cal = _nn_dist(m, Xtr, cal)
    d_tgt = _nn_dist(m, Xtr, tgt)
    p_cal_full = MD._batched_p(m, calib_b)
    p_tgt = MD._batched_p(m, tgt)
    res = {"n_train_benign": int(len(train_b)), "n_calib": int(len(cal)), "n_target": int(len(tgt)),
           "nn_distance_target_mean": float(d_tgt.mean()), "nn_distance_target_median": float(np.median(d_tgt)),
           "nn_distance_calib_median": float(np.median(d_cal)), "reference_alert_rate": reference}
    for a in ALPHAS:
        r = float(np.quantile(d_cal, 1 - a))
        thr = float(np.quantile(p_cal_full, 1 - a, method="higher"))
        eps = float((d_tgt > r).mean())
        alert = float((p_tgt >= thr).mean())
        # alert rate among the target rows inside the radius, where Proposition 3 expects few alerts
        inside = d_tgt <= r
        viol = float((p_tgt[inside] >= thr).mean()) if inside.any() else None
        res[f"alpha_{a}"] = {"radius": r, "threshold": thr, "eps": eps, "alert_rate_on_sample": alert,
                             "alert_rate_inside_radius": viol, "bound_holds": bool(alert <= eps + RHO)}
    a1 = res["alpha_0.01"]
    print(f"  eps@1%={a1['eps']:.4f} alert@1%={a1['alert_rate_on_sample']:.4f} reference {reference}, "
          f"alert rate inside the radius {a1['alert_rate_inside_radius']}  [{time.time() - t0:.0f}s]", flush=True)
    return res


def main():
    t0 = time.time()
    os.makedirs(MDIR, exist_ok=True)
    out = {"config": {"seed": C.SEED, "n_target": N_TARGET, "n_calib": N_CALIB, "alphas": ALPHAS,
                      "representation": "TextOnly char 3-5 TF-IDF, two fields, L2-normalised; cosine distance"}}
    atk = MD.build_attacks()
    EXT = MD.EXT
    cols = ["timestamp", "http_method", "url_path_raw", "url_query", "request_body", "headers", "label"]

    # in-domain and cross-domain pairs, on the same splits as multidomain and weblog_edgar
    zan = pd.read_parquet(f"{EXT}/zanbil.parquet", columns=[c for c in cols if c != "request_body"])
    zan = zan[zan.label == "benign"]; zan["request_body"] = ""
    edg = pd.read_parquet(f"{EXT}/edgar.parquet", columns=cols); edg = edg[edg.label == "benign"]
    wl = pd.read_parquet(f"{EXT}/weblog2025.parquet"); wlb = wl[wl.label == "benign"].copy()
    srb = pd.read_parquet(f"{EXT}/srbh.parquet", columns=cols); srb = srb[srb.label == "benign"]

    ztr, zca, zte, _ = MD.split_benign(zan, "temporal")
    out["zanbil_in_domain"] = measure("zanbil_in_domain", ztr, zca, zte, atk["train"], 0.00446, t0)
    # cross-domain: the same zanbil training set, scored on EDGAR and WebLog benign
    out["zanbil_to_edgar"] = measure("zanbil_to_edgar", ztr, zca, edg, atk["train"], 1.0, t0)
    # reference: the Zanbil model at its own 1% threshold on all WebLog benign (weblog_edgar.json weblog_alert_rate@own_1pct)
    out["zanbil_to_weblog"] = measure("zanbil_to_weblog", ztr, zca, wlb, atk["train"], 0.96986, t0)
    etr, eca, ete, _ = MD.split_benign(edg, "temporal")
    out["edgar_in_domain"] = measure("edgar_in_domain", etr, eca, ete, atk["train"], 0.00608, t0)
    wtr, wca, wte, _ = MD.split_benign(wlb, "temporal")
    out["weblog_temporal"] = measure("weblog_temporal", wtr, wca, wte, atk["train"], 0.20024, t0)
    htr, hca, hte, _ = MD.split_benign(wlb, "hash")
    out["weblog_hash"] = measure("weblog_hash", htr, hca, hte, atk["train"], 0.0099, t0)
    str_, sca, ste, _ = MD.split_benign(srb, "temporal")
    out["srbh_in_domain"] = measure("srbh_in_domain", str_, sca, ste, atk["train"], 0.5346, t0)

    # leave-one-source-out folds, assembled exactly as loso_eval does
    all_src = LO._load_all()
    print("sources loaded:", sorted(all_src))
    for bstar, astar in LO.FOLDS:
        need = [bstar, astar[0] if isinstance(astar, tuple) else astar]
        missing = [n for n in need if n not in all_src]
        assert not missing, f"source not loaded: {missing}; run scripts/fetch_data.sh"
        fit, calib, Bstar, _Astar, _meta = LO.assemble(all_src, bstar, astar)
        tb = fit[fit.label == "benign"]; ta = fit[fit.label == "attack"]; cb = calib[calib.label == "benign"]
        key = f"loso_{bstar}_heldout" + ("_srbh_in_train" if isinstance(astar, tuple) else "")
        out[key] = measure(key, tb, cb, Bstar, ta, _REF_LOSO[(bstar, astar)], t0)

    with open(os.path.join(MDIR, "representation_coverage_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"done  [{time.time() - t0:.0f}s]")
    return out


if __name__ == "__main__":
    main()
