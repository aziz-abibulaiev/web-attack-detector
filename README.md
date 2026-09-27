# Web Attack Detector

This repository contains the code and data pipeline behind *Limits of Web Attack Detection in
Real-World Production Environments*, by Abibulaiev A., Pukach P. and Vovk M., submitted to MDPI
*Applied Sciences* in 2026. It downloads the public datasets, trains and evaluates the detector,
writes the result tables, and checks them against the reference values in `expected_metrics/`.
Nothing outside this repository is needed.

## Requirements

- macOS or Linux, CPU only. The reference run is macOS on an Apple M3 Pro, 11 cores.
- Python 3.13.7, with the pinned versions in `env/requirements.lock`.
- 16 GB of disk after the download, about 16 GB of RAM, and about 85 minutes for the reproduction
  itself. Only 0.6 GB of the downloaded data is needed to run it; the rest of `data/raw/` can be
  deleted once the parquets exist.
- `curl`, `unzip` and `gzip`; the Kaggle CLI with credentials in `~/.kaggle/kaggle.json`, for
  zanbil and CSIC; `zenodo_get` for WebLog-2025 and ModSecurity, from `pip install zenodo_get`.
- A `SEC_USER_AGENT` environment variable for the SEC EDGAR logs. The SEC Fair Access policy
  requires automated clients to identify themselves with an organization and a contact address.

## Setup

1. Clone the repository.

```bash
git clone https://github.com/aziz-abibulaiev/web-attack-detector.git && cd web-attack-detector
```

2. Create the virtual environment and install the pinned dependencies.

```bash
python3.13 -m venv .venv && source .venv/bin/activate && pip install -r env/requirements.lock
```

3. Create a Kaggle API token with "Create New Token" in the Kaggle account settings, save it to
   `~/.kaggle/kaggle.json`, then restrict its permissions.

```bash
chmod 600 ~/.kaggle/kaggle.json
```

4. Export a contact address for the SEC downloads.

```bash
export SEC_USER_AGENT="Your Organization you@example.org"
```

## Getting the data

```bash
bash scripts/fetch_data.sh
```

The script downloads the raw archives into `data/raw/`, builds one parquet per dataset in the
unified schema under `data/corpus/external/`, and downloads the frozen testbed corpus into
`data/corpus/labelled/`. After each dataset it asserts the exact row count listed in the dataset
table below, and a mismatch stops the run. It is safe to re-run: an archive already on disk is
not downloaded again and a parquet already built is not rebuilt, but every count assert runs
again. `bash scripts/fetch_data.sh weblog edgar` fetches only the named datasets.

## Running the reproduction

```bash
bash scripts/reproduce.sh
```

The script runs 15 steps and verifies the results. The order is fixed: several steps load a model
an earlier step wrote.

| step | what it computes | output | paper |
|---|---|---|---|
| `lab_testbed` | detector trained and evaluated on the testbed corpus: binary core, cross-tool holdout, marker-free recall, family head | `models/lab_testbed/eval_results.json` | 7.5, 7.6 |
| `lab_external` | that testbed-trained detector scored on external traffic: zanbil sample, SR-BH benign, SR-BH attacks | `models/lab_testbed/lab_external_results.json` | Table 6 |
| `single_source` | one real benign source in training, alert rate on a different one | `models/single_source/single_source_results.json` | 7.1 |
| `loso` | leave-one-source-out transfer across benign and attack sources | `models/loso/loso_results.json` | 7.1 |
| `in_domain` | the in-domain operating point on zanbil | `models/in_domain/in_domain_results.json` | 4.5, 7.2 |
| `ablation_positives` | the in-domain zanbil model retrained with content families only among the positives | `models/in_domain/ablation_positives_results.json` | 5.9.1 |
| `ml_vs_rules` | learned detector against the regex baseline, cross-tool recall, numeric-channel ablation | `models/in_domain/ml_vs_rules_results.json` | 7.3-7.5 |
| `multidomain` | the same in-domain protocol on zanbil and SR-BH | `models/multidomain/multidomain_results.json` | 7.2 |
| `adversarial` | recall under static WAF-bypass mutations | `models/adversarial/adversarial_results.json` | 7.7 |
| `weblog_edgar` | WebLog-2025 and SEC EDGAR as further domains | `models/weblog_edgar/weblog_edgar_results.json` | 7.2 |
| `multisite` | WebLog subdomains: hash, temporal and site-stratified splits | `models/weblog_edgar/multisite_coverage_results.json` | 4.5, 7.2 |
| `coverage` | representation coverage: nearest-neighbor distance to the training benign set, for every training and target pair | `models/coverage/representation_coverage_results.json` | 7.8 |
| `sample` | rebuilds the 100-row SR-BH family sample from the stored labels | `data/corpus/samples/srbh_family_sample.parquet` | input to `family` |
| `family` | the family head on author-verified real attacks | `models/in_domain/family_results.json` | 7.6 |
| `figure` | the main result figure | `figures/main_result.pdf`, `.png` | Figure 1 |

`FAST=1` skips the six slowest steps, `lab_testbed`, `lab_external`, `single_source`, `loso`,
`ablation_positives` and `coverage`, and verifies the rest in about 30 minutes.
`STEPS="in_domain family"` runs only those steps.
The run ends by calling `scripts/check_metrics.py`, which prints:

```
table                    result   ndiff  detail
-----------------------------------------------------
lab_testbed              PASS        0
lab_external             PASS        0
single_source_fail       PASS        0
loso                     PASS        0
in_domain                PASS        0
ml_vs_rules              PASS        0
family                   PASS        0
multidomain              PASS        0
adversarial              PASS        0
weblog_edgar             PASS        0
multisite_coverage       PASS        0
representation_coverage  PASS        0
ablation_positives       PASS        0
-----------------------------------------------------
13/13 PASS  (tol=1e-06)
```

If a check fails, the script prints the JSON path of each mismatching value with the expected
value, the reproduced value and the difference.

## Repository layout

```
src/content_detector/   the detector and the 15 evaluation drivers
src/corpus/             the tooling that captured the testbed corpus
scripts/                fetch_data.sh, reproduce.sh, check_metrics.py, marker_free_sample.py
expected_metrics/       the reference result files the reproduction is checked against
models/in_domain/       family_labels.csv, the stored annotation record
docs/corpus/manifests/  the capture manifests of the frozen testbed corpus
docs/review/            the marker-free sample of Section 5.9.1, classified by hand
env/                    requirements.lock, the pinned environment
data/, models/, figures/  created by the scripts, not tracked in git
```

## Datasets

Every dataset is public. The counts are what `fetch_data.sh` asserts, after deduplication.

| id | source | coordinates | checksum | asserted count |
|---|---|---|---|---|
| `zanbil` | e-commerce nginx access log, Iran 2019 | Kaggle `eliasdabbas/web-server-access-logs` | row count only | 893,936 benign |
| `edgar` | SEC EDGAR filing-server logs, first 1.2 M rows of 2015-06-01, 2016-03-01, 2017-05-02 | `www.sec.gov/dera/data/Public-EDGAR-log-file-data` | row count only | 1,661,635 benign |
| `weblog2025` | multi-tenant WordPress host, 22 sites, 2025 | Zenodo `10.5281/zenodo.20001206` | zip MD5 `b1925ff7f043c391d501cfd404e0df10` | 757,845 benign, 129 attack |
| `srbh` | SR-BH 2020 WordPress honeypot | Harvard Dataverse `doi:10.7910/DVN/OGOIXX` | csv MD5 `173ec515308bdce5aec19cfd5b792596` | 90,917 benign, 287,710 attack |
| `nasa`, `clarknet`, `calgary` | Internet Traffic Archive, 1990s | `ita.ee.lbl.gov/traces` | row count only | 8,436 / 18,411 / 8,152 benign |
| `csic` | CSIC 2010 | Kaggle `ispangler/csic-2010-web-application-attacks` | row count only | 9,644 benign, 15,954 attack |
| `modsec2025` | requests blocked by OWASP ModSecurity on a production server | Zenodo `10.5281/zenodo.17178461` | zip MD5 `95b7a8237abc163d8ca31e49f7318efd` | 15,262 attack |
| `testbed` | crAPI and VAmPI captured with real tools | release asset below | sha256 below | 10,908 benign, 10,228 attack |

The testbed corpus is shipped frozen, because a live capture is not bit-reproducible:

```
https://github.com/aziz-abibulaiev/web-attack-detector/releases/download/v1.0.0/testbed_corpus.tar.gz
sha256 9126e43da413bee16cb543a8d0b9187819d312303f83c83e57d8bda3b1ecfd86
```

Its labels come only from each capture campaign's source IP and time window, never from request
content. The manifests are in `docs/corpus/manifests/`, and `src/corpus/README.md` describes the
capture tooling. `docs/review/` holds 100 marker-free testbed attacks classified by hand for
Section 5.9.1, and `scripts/marker_free_sample.py` redraws the unclassified sample.

Two dataset families are not used. Biblio-US17 is the best human-verified real dataset, but access
is gated behind a login and a CAPTCHA. CICIDS2017, CSE-CIC-IDS2018 and UNSW-NB15 are flow-level
and carry no request payload, so a content detector cannot be evaluated on them. Every alert rate
measured on real traffic is an upper bound on the false-positive rate, because real benign logs
contain undetected attacks.

## Determinism and tolerance

Seed 42 fixes every split and every sample, `reproduce.sh` exports `PYTHONHASHSEED=0`, and the
estimators are deterministic. In the pinned environment the reproduction returns 12 of the 13
result files byte for byte; `multisite_coverage` differs in one quantile threshold by about
7e-16, so `check_metrics.py` passes at its default `--tol 1e-6`. On a different scikit-learn or
BLAS build, values can drift by about 1e-4 from solver and reduction-order differences; use
`--tol 1e-3`.

## Citation and license

Abibulaiev A., Pukach P., Vovk M. *Limits of Web Attack Detection in Real-World Production
Environments.* Submitted to MDPI Applied Sciences, 2026.

The code is MIT licensed, see `LICENSE`. Each dataset keeps its own upstream license; the only
data this repository distributes is the frozen testbed corpus, which is the authors' own capture.
