# Marker-free sample

`marker_free_sample_classified.csv` holds 100 of the 6,507 testbed attacks that match no pattern
of the rule dictionary, stratified by family with seed 42. `sample_idx` is the position in the
sample, `testbed_row` the row of the attack in `data/corpus/labelled/testbed.parquet`, and the
remaining columns are the request fields. The `group` column is the authors' classification by
hand into the three groups of Section 5.9.1: `content_payload_unmatched` is a content-visible
payload the dictionary does not match (group a, 26 rows), `tool_request_no_payload` is a tool
request without a payload in the content (group b, 17 rows), and `outside_content_channel` is a
request whose attack character lies outside the content channel (group c, 57 rows). No script
computes that column.

To regenerate the unclassified sample, the same 100 rows in the same order, as
`data/corpus/samples/marker_free_sample.csv`, with the per-family counts printed:

```bash
python scripts/marker_free_sample.py
```
