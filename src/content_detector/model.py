"""ContentDetector: a character n-gram TF-IDF text channel and a shape-feature numeric channel
combined by late fusion, with a binary attack/benign core and an optional family head.
Hyperparameters come from config.py; calibration and the operating threshold are fit only on a
calibration slice that does not overlap test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config as C
from .features import FEATURE_NAMES, numeric_features
from .normalize import assemble_fields

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class ContentDetector:
    def __init__(self):
        self.vec_a = TfidfVectorizer(max_features=C.TFIDF_MAX_FEATURES_A, **C.TFIDF)
        self.vec_b = TfidfVectorizer(max_features=C.TFIDF_MAX_FEATURES_B, **C.TFIDF)
        self.text_clf = LogisticRegression(**C.LOGREG)
        self.num_clf = HistGradientBoostingClassifier(**C.HGB)
        self.platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)  # 1-D calibration
        self.thresholds: dict[float, float] = {}
        self.fitted = False

    # ---- feature builders ----
    @staticmethod
    def _fields(reqs: list[dict]):
        fa, fb = [], []
        for r in reqs:
            f = assemble_fields(url_path_raw=r.get("url_path_raw", r.get("path", "")),
                                url_query=r.get("url_query", r.get("query", "")),
                                request_body=r.get("request_body", r.get("body", "")),
                                headers=r.get("headers"))
            fa.append(f["field_a"])
            fb.append(f["field_b"])
        return fa, fb

    @staticmethod
    def _numeric_matrix(reqs: list[dict]) -> pd.DataFrame:
        rows = [numeric_features(url_path_raw=r.get("url_path_raw", r.get("path", "")),
                                 url_query=r.get("url_query", r.get("query", "")),
                                 request_body=r.get("request_body", r.get("body", "")),
                                 headers=r.get("headers")) for r in reqs]
        return pd.DataFrame(rows, columns=FEATURE_NAMES)

    def _text_matrix(self, reqs, fit=False):
        fa, fb = self._fields(reqs)
        if fit:
            Xa = self.vec_a.fit_transform(fa)
            Xb = self.vec_b.fit_transform(fb)
        else:
            Xa = self.vec_a.transform(fa)
            Xb = self.vec_b.transform(fb)
        return hstack([Xa, Xb]).tocsr()

    # ---- channels ----
    def p_text(self, reqs):
        return self.text_clf.predict_proba(self._text_matrix(reqs))[:, 1]

    def p_numeric(self, reqs):
        return self.num_clf.predict_proba(self._numeric_matrix(reqs))[:, 1]

    def _fuse(self, pt, pn):
        wt, wn = C.FUSION_WEIGHTS
        return wt * _logit(pt) + wn * _logit(pn)  # fused logit

    def p_content(self, reqs, pt=None, pn=None):
        pt = self.p_text(reqs) if pt is None else pt
        pn = self.p_numeric(reqs) if pn is None else pn
        fused = self._fuse(pt, pn)
        return self.platt.predict_proba(fused.reshape(-1, 1))[:, 1]

    # ---- fit ----
    def fit(self, train_reqs, y_train, calib_reqs, y_calib):
        y_train = np.asarray(y_train, dtype=int)
        y_calib = np.asarray(y_calib, dtype=int)
        # text channel
        Xt = self._text_matrix(train_reqs, fit=True)
        self.text_clf.fit(Xt, y_train)
        # numeric channel
        self.num_clf.fit(self._numeric_matrix(train_reqs), y_train)
        # fusion calibration on CALIB only
        pt_c = self.p_text(calib_reqs)
        pn_c = self.p_numeric(calib_reqs)
        fused_c = self._fuse(pt_c, pn_c)
        self.platt.fit(fused_c.reshape(-1, 1), y_calib)
        # FPR-budget thresholds from CALIB benign
        pc_c = self.platt.predict_proba(fused_c.reshape(-1, 1))[:, 1]
        benign = pc_c[y_calib == 0]
        for a in C.FPR_BUDGETS:
            self.thresholds[a] = float(np.quantile(benign, 1 - a, method="higher")) if len(benign) else 0.5
        self.fitted = True
        return self

    def threshold(self, fpr=None):
        return self.thresholds[fpr or C.PRIMARY_FPR]
