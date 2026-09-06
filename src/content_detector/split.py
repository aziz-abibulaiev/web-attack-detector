"""Splits the testbed corpus by session and stratifies by family, never by row. Each generator
used one source IP per tool, so every content-injection attack sits in one session per tool:
putting the ffuf and sqlmap sessions in train and the custom session in test makes the main
split a cross-tool split. Behavioral and context families are distributed normally and benign
sessions are hashed 60/20/20. Asserts that no session_id appears in two splits.
"""

from __future__ import annotations

import hashlib

from .config import BEHAVIOURAL_FAMILIES, CONTENT_FAMILIES, CONTEXT_FAMILIES, SEED

# deterministic tool assignment for the 3 content-injection sessions
_CONTENT_TOOL_SPLIT = {"ffuf": "train", "sqlmap": "train", "custom": "test"}
# one context family per split: each has about one real session, too few to stratify within
_CONTEXT_SPLIT = {"bola_idor": "train", "mass_assignment": "calib", "business_logic_abuse": "test"}


def _hash_bucket(key: str, train=0.6, calib=0.2) -> str:
    h = int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest(), 16) % 10_000 / 10_000.0
    if h < train:
        return "train"
    if h < train + calib:
        return "calib"
    return "test"


def assign_splits(df) -> dict:
    """Return {session_id: 'train'|'calib'|'test'}."""
    assign: dict[str, str] = {}
    # summarize each session
    sess = df.groupby("session_id").agg(
        label=("label", lambda s: s.iloc[0]),
        fams=("attack_family", lambda s: set(x for x in s.dropna().unique())),
        tool=("attack_tool", lambda s: next((x for x in s.dropna().unique()), None)),
    )

    # behavioral: stratify the 10 sessions/family into 6 train / 2 calib / 2 test
    behav_by_fam: dict[str, list] = {f: [] for f in BEHAVIOURAL_FAMILIES}

    for sid, row in sess.iterrows():
        fams = row["fams"]
        if row["label"] == "benign" or not fams:
            assign[sid] = _hash_bucket(sid)
            continue
        fam = sorted(fams)[0] if len(fams) == 1 else None
        # content-injection sessions: assign by tool, which makes the split cross-tool
        if fams <= set(CONTENT_FAMILIES):
            assign[sid] = _CONTENT_TOOL_SPLIT.get(row["tool"], "train")
        elif fam in BEHAVIOURAL_FAMILIES:
            behav_by_fam[fam].append(sid)
        elif fam in CONTEXT_FAMILIES:
            assign[sid] = _CONTEXT_SPLIT.get(fam, "train")
        else:
            assign[sid] = _hash_bucket(sid)

    for fam, sids in behav_by_fam.items():
        sids = sorted(sids)  # deterministic
        n = len(sids)
        n_tr = round(n * 0.6)
        n_ca = round(n * 0.2)
        for i, sid in enumerate(sids):
            assign[sid] = "train" if i < n_tr else ("calib" if i < n_tr + n_ca else "test")

    return assign


def split_frame(df):
    """Attach a `_split` column, assert zero session overlap, return the frame and its stats."""
    assign = assign_splits(df)
    df = df.copy()
    df["_split"] = df["session_id"].map(assign)

    sids = {s: set(df.loc[df._split == s, "session_id"]) for s in ("train", "calib", "test")}
    overlaps = {}
    names = list(sids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sids[a] & sids[b]
            if inter:
                overlaps[f"{a}∩{b}"] = sorted(inter)
    assert not overlaps, f"session overlap across splits: {overlaps}"

    stats = {}
    for s in ("train", "calib", "test"):
        part = df[df._split == s]
        fam_counts = part[part.label == "attack"]["attack_family"].value_counts().to_dict()
        stats[s] = {
            "rows": int(len(part)), "sessions": int(part.session_id.nunique()),
            "benign": int((part.label == "benign").sum()),
            "attack": int((part.label == "attack").sum()),
            "attack_by_family": fam_counts,
        }
    return df, {"splits": stats, "session_overlaps": overlaps, "disjoint": not overlaps}
