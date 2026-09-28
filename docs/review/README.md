# Marker-free sample

`marker_free_sample_classified.csv` holds 100 of the 6,507 testbed attacks that match no pattern
of the rule dictionary, stratified by family with seed 42. `sample_idx` is the position in the
sample, `testbed_row` the row of the attack in `data/corpus/labelled/testbed.parquet`, and the
remaining columns are the request fields. The `group` column is the authors' manual classification
into the three groups of Section 5.9.1: `content_payload_unmatched` is a content-visible
payload the dictionary does not match (group a, 26 rows), `tool_request_no_payload` is a tool
request without a payload in the content (group b, 17 rows), and `outside_content_channel` is a
request whose attack character lies outside the content channel (group c, 57 rows). No script
computes that column.

To regenerate the unclassified sample, the same 100 rows in the same order, as
`data/corpus/samples/marker_free_sample.csv`, with the per-family counts printed:

```bash
python scripts/marker_free_sample.py
```

# Highest-scoring Zanbil alerts

`top100_zanbil_classified.csv` holds the 100 highest-scoring alerts of the in-domain Zanbil model on
the later Zanbil benign window, ordered by score. `rank` is the position by score, `score` the
detector output, `regex_marked` whether the audit expression of `in_domain_train.py` matches the
undecoded request, and the remaining columns are the request fields. The `class` and `kind` columns
are the authors' manual classification of each request for Section 7.2: `attack` with a marker is a
request the audit expression matches (31 rows: 25 SQL injection, 5 path traversal or file inclusion,
1 command injection), `attack` without a marker is an attack the expression misses (30 rows: 15 SQL
injection, 10 probes for Magento files, 3 open-proxy requests, 2 exploit probes), and `benign` is an
ordinary request of the site (39 rows). No script computes those columns. The key
`wamm_manual.true_fp_estimate` of `expected_metrics/in_domain.json` counts the rows without a marker
and is a diagnostic of the audit expression, not the reported result.

To regenerate the unclassified list, the same 100 rows in the same order with the `regex_marked`
column, as `models/in_domain/manual_inspect_late_zanbil.csv`:

```bash
STEPS="in_domain" bash scripts/reproduce.sh
```
