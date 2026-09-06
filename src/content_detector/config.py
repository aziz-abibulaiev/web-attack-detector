"""Hyperparameters, sample sizes, seeds and output directories for the content detector.
Every value is set here and used unchanged by the drivers; nothing in the pipeline searches
or tunes them.
"""

SEED = 42

# ---- shared normalization ----
DECODE_MAX_PASSES = 3  # unquote_plus to a fixed point, capped; catches double encoding

# ---- p_text: char n-gram TF-IDF -> linear ----
# Two SEPARATE text fields, each its own vectorizer, so their weights differ:
#   field A = decode_fixed_point(url_path_raw + url_query + request_body)
#   field B = decode_fixed_point(user_agent + referer)
TFIDF = dict(analyzer="char", ngram_range=(3, 5), min_df=2, lowercase=False, sublinear_tf=True)
# max_features caps memory; generous and fixed, not tuned.
TFIDF_MAX_FEATURES_A = 200_000
TFIDF_MAX_FEATURES_B = 50_000
LOGREG = dict(class_weight="balanced", max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)

# ---- p_numeric: shape statistics -> gradient boosting ----
HGB = dict(
    learning_rate=0.1, max_iter=300, max_depth=None, max_leaf_nodes=31,
    l2_regularization=1.0, early_stopping=False, random_state=SEED,
)

# ---- fusion ----
# Mean of the logits of p_text and p_numeric, then a 1-D Platt calibration fit on the
# calibration slice alone, so each channel's contribution stays measurable.
FUSION = "mean_logit"
FUSION_WEIGHTS = (0.5, 0.5)  # text weight, then numeric

# ---- operating point ----
# FPR-budget threshold chosen on the CALIBRATION slice only; never on test.
FPR_BUDGETS = (0.01, 0.05)
PRIMARY_FPR = 0.01

# ---- family head, testbed only ----
HGB_FAMILY = dict(
    learning_rate=0.1, max_iter=300, max_depth=None, max_leaf_nodes=31,
    l2_regularization=1.0, early_stopping=False, random_state=SEED,
)

# cross-tool families with zero payload overlap between custom and ffuf
CLEAN_XTOOL_FAMILIES = ["sqli", "xss", "cmd_injection", "ssti"]
# families whose relabeled ffuf slice is off-family content -> binary only, no family holdout
BINARY_ONLY_FFUF_OFFCONTENT = ["nosql_injection", "ssrf"]
# families with a small shared-payload intersection to drop before per-family holdout
DROP_SHARED_BEFORE_XTOOL = ["path_traversal", "scan"]

CONTENT_FAMILIES = ["sqli", "xss", "cmd_injection", "nosql_injection", "ldap_injection",
                    "ssti", "path_traversal", "ssrf", "scan"]
BEHAVIOURAL_FAMILIES = ["brute_force", "cred_stuffing", "resource_exhaustion"]
CONTEXT_FAMILIES = ["bola_idor", "mass_assignment", "business_logic_abuse"]

MODEL_DIR_LAB_TESTBED = "models/lab_testbed"

# ---- single-source real-benign training; the sizes are declared, not tuned ----
MODEL_DIR_SINGLE_SOURCE = "models/single_source"
# real benign is deduplicated on the request signature, then hash-split by that key so
# train/calib never share a row. The training source is never the FPR-test source.
# ---- leave-one-source-out across several sources ----
MODEL_DIR_LOSO = "models/loso"
PER_SOURCE_BENIGN_CAP = 30_000         # cap per benign source in train, to bound the TF-IDF
PER_SOURCE_ATTACK_CAP = 30_000         # cap per attack source in train
LOSO_CALIB_FRAC = 0.15                 # fraction of assembled real-benign train held for calib
LOSO_BSTAR_EVAL = 80_000               # held-out benign source rows sampled for FP
LOSO_ASTAR_EVAL = 40_000               # held-out attack rows sampled for recall

REAL_BENIGN_TRAIN_FRAC = 0.7            # remainder -> calib; the FPR test is the other source
ZANBIL_TRAIN_SAMPLE = 140_000          # unique zanbil rows drawn for A2 train+calib
ZANBIL_FPR_TEST_SAMPLE = 200_000       # unique zanbil rows for A1 held-out FPR test
SRBH_ATTACK_EVAL_SAMPLE = 100_000      # SR-BH attacks sampled for recall
MANUAL_INSPECT_N = 100                 # high-confidence alerts on held-out benign to eyeball
WAMM_HIDDEN_ATTACK_RATE = 0.095        # ~9-10% of SR-BH "benign" are real attacks, and stay in
