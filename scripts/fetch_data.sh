#!/usr/bin/env bash
# Downloads the public datasets, builds one unified-schema parquet per dataset, and asserts the
# row counts listed in the Datasets section of the README. A mismatch stops the run.
#
# Re-running is safe: an archive already on disk is not downloaded again, and neither is a parquet
# already built rebuilt, but every count assert runs again. Checksums are verified where known.
# Delete a parquet to force its rebuild. The frozen testbed corpus comes from a GitHub release
# asset; nothing here reads a file outside this repository.
#
# Requirements: curl, unzip, gzip, the pinned venv from env/requirements.lock, the Kaggle CLI with
# ~/.kaggle/kaggle.json for zanbil and csic; zenodo_get for weblog and modsec;
# SEC_USER_AGENT="Your Organization you@example.org" for edgar, per SEC Fair Access policy.
# Usage:  bash scripts/fetch_data.sh                # everything
#         bash scripts/fetch_data.sh weblog edgar   # a subset
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="${PYTHON:-python}"
EXT="data/corpus/external"; LAB="data/corpus/labelled"; RAW="data/raw"
mkdir -p "$EXT/ita" "$LAB" "$RAW"
CORPUS_RELEASE_URL="${CORPUS_RELEASE_URL:-https://github.com/aziz-abibulaiev/web-attack-detector/releases/download/v1.0.0/testbed_corpus.tar.gz}"
CORPUS_SHA256="${CORPUS_SHA256:-9126e43da413bee16cb543a8d0b9187819d312303f83c83e57d8bda3b1ecfd86}"

say(){ printf "\n\033[1m== %s\033[0m\n" "$*"; }
die(){ printf "\033[31m!! %s\033[0m\n" "$*" >&2; exit 1; }
have(){ if [ -f "$1" ]; then echo "   [skip] $1 already present"; return 0; fi; return 1; }
sha256_of(){ shasum -a 256 "$1" | awk '{print $1}'; }
md5_of(){ if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | awk '{print $1}'; fi; }
# dl <url> <out> [extra curl args…] — retries, and aborts a transfer that stalls under 1 kB/s for 60 s
dl(){ url="$1"; out="$2"; shift 2
  curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 --speed-limit 1024 --speed-time 60 "$@" "$url" -o "$out"; }

# assert_rows <parquet> <expected_rows> [<expected_benign>] — rows of a built parquet
assert_rows(){ $PY - "$@" <<'PY' || die "row-count assertion FAILED for $1 (see the README)"
import sys, pandas as pd
p=sys.argv[1]; exp=int(sys.argv[2])
df=pd.read_parquet(p, columns=["label"]) if len(sys.argv)>3 else pd.read_parquet(p)
ok=len(df)==exp; print(f"[assert] {p}: rows={len(df)} expected={exp} -> {'OK' if ok else 'MISMATCH'}")
if len(sys.argv)>3:
    nb=int((df.label=="benign").sum()); okb=nb==int(sys.argv[3]); ok=ok and okb
    print(f"[assert] {p}: benign={nb} expected={sys.argv[3]} -> {'OK' if okb else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PY
}
# assert_loader <name> <expected_rows> [<expected_benign>] — what the driver actually loads,
# after its own dedup, via src.content_detector.sources.load_<name>()
assert_loader(){ $PY - "$@" <<'PY' || die "loader-count assertion FAILED for $1 (see the README)"
import sys; sys.path.insert(0, ".")
from src.content_detector import sources as S
name=sys.argv[1]; exp=int(sys.argv[2]); df=getattr(S, f"load_{name}")()
ok=len(df)==exp; print(f"[assert] load_{name}(): rows={len(df)} expected={exp} -> {'OK' if ok else 'MISMATCH'}")
if len(sys.argv)>3:
    nb=int((df.label=="benign").sum()); okb=nb==int(sys.argv[3]); ok=ok and okb
    print(f"[assert] load_{name}(): benign={nb} expected={sys.argv[3]} -> {'OK' if okb else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PY
}
zenodo_fetch(){ # $1 record id  $2 dest dir  $3 expected file
  mkdir -p "$2"; have "$2/$3" && return 0
  command -v zenodo_get >/dev/null 2>&1 || die "zenodo_get not installed (pip install zenodo_get)"
  ( cd "$2" && zenodo_get "10.5281/zenodo.$1" ) || die "zenodo record $1"
}
kaggle_fetch(){ # $1 dataset  $2 dest dir  $3 expected file
  have "$2/$3" && return 0
  command -v kaggle >/dev/null 2>&1 || die "kaggle CLI required (pip install kaggle; ~/.kaggle/kaggle.json)"
  kaggle datasets download -d "$1" -p "$2" --unzip || die "kaggle $1"
}

fetch_zanbil(){ say "zanbil — Kaggle eliasdabbas/web-server-access-logs, a real e-commerce nginx log"
  kaggle_fetch eliasdabbas/web-server-access-logs "$RAW/zanbil" access.log
  have "$EXT/zanbil.parquet" || $PY -m src.corpus.ingest.zanbil_ingest --log "$RAW/zanbil/access.log" --out "$EXT/zanbil.parquet"
  assert_loader zanbil 893936                                   # unique benign after the driver's dedup
}
fetch_srbh(){ say "srbh — Harvard Dataverse doi:10.7910/DVN/OGOIXX, the SR-BH 2020 honeypot"
  csv=$(find "$RAW/srbh" -iname '*.csv' 2>/dev/null | head -1)
  if [ -z "$csv" ]; then
    dl "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/OGOIXX" \
       "$RAW/srbh_dataverse.zip" || die "dataverse srbh download"
    mkdir -p "$RAW/srbh" && unzip -o -q "$RAW/srbh_dataverse.zip" -d "$RAW/srbh"
    csv=$(find "$RAW/srbh" -iname '*.csv' | head -1); [ -n "$csv" ] || die "SR-BH csv not found in the Dataverse bundle"
  else echo "   [skip] $csv already present"; fi
  got=$(md5_of "$csv"); [ "$got" = "173ec515308bdce5aec19cfd5b792596" ] || echo "   [warn] SR-BH csv md5 $got differs from the documented 173ec515…; the row-count assert below is the contract"
  have "$EXT/srbh.parquet" || $PY src/corpus/ingest/srbh_ingest.py --csv "$csv" --out "$EXT/srbh.parquet"
  assert_rows "$EXT/srbh.parquet" 907815                       # raw rows written by the ingester
  assert_loader srbh 378627 90917                             # after the driver's dedup: 90,917 benign / 287,710 attack
}
fetch_edgar(){ say "edgar — SEC EDGAR log files: 2015-06-01, 2016-03-01, 2017-05-02"
  # SEC Fair Access policy: automated clients must declare a User-Agent of the form
  # "Organization contact@email", stay well under 10 req/s, and HEAD is always refused. Repeated
  # non-compliant requests get the IP blocked for about ten minutes. One GET at a time, https.
  : "${SEC_USER_AGENT:?set SEC_USER_AGENT=\"Your Organization you@example.org\" — www.sec.gov Fair Access policy: automated clients must declare a contact in the User-Agent}"
  base="https://www.sec.gov/dera/data/Public-EDGAR-log-file-data"; csvs=""
  for rel in 2015/Qtr2/log20150601.zip 2016/Qtr1/log20160301.zip 2017/Qtr2/log20170502.zip; do
    zip="$RAW/$(basename "$rel")"; csv="$RAW/edgar/$(basename "${rel%.zip}").csv"
    have "$zip" || { dl "$base/$rel" "$zip" --retry-delay 20 -A "$SEC_USER_AGENT" \
                        -H "Accept-Encoding: gzip, deflate" -H "Host: www.sec.gov" || die "edgar $rel"; sleep 1; }
    have "$csv" || unzip -o -q "$zip" -d "$RAW/edgar"
    csvs="$csvs $csv"
  done
  # --edgar-nrows 1200000 takes the first 1.2M rows of each day: a fixed prefix, not a sample
  have "$EXT/edgar.parquet" || $PY -m src.content_detector.ingest_weblog_edgar --edgar-csv $csvs --edgar-nrows 1200000 --edgar-out "$EXT/edgar.parquet"
  assert_rows "$EXT/edgar.parquet" 1661635
}
fetch_weblog(){ say "weblog2025 — Zenodo 10.5281/zenodo.20001206, a multi-tenant WordPress host"
  zenodo_fetch 20001206 "$RAW/weblog" organization-y.zip
  [ "$(md5_of "$RAW/weblog/organization-y.zip")" = "b1925ff7f043c391d501cfd404e0df10" ] || die "organization-y.zip md5 mismatch (Zenodo publishes b1925ff7…)"
  [ -d "$RAW/weblog/organization-y" ] || ( cd "$RAW/weblog" && unzip -o -q organization-y.zip )
  have "$EXT/weblog2025.parquet" || $PY -m src.content_detector.ingest_weblog_edgar \
      --weblog "$RAW"/weblog/organization-y/log/home/*/logs/*access_log* \
      --weblog-yaml "$RAW/weblog/organization-y/ground-truth/organization-y.yaml" --weblog-out "$EXT/weblog2025.parquet"
  assert_rows "$EXT/weblog2025.parquet" 757974 757845            # 757,845 benign / 129 attack
}
fetch_modsec(){ say "modsec2025 — Zenodo 10.5281/zenodo.17178461, owasp.zip, real WAF-flagged requests"
  zenodo_fetch 17178461 "$RAW/modsec" owasp.zip
  [ "$(md5_of "$RAW/modsec/owasp.zip")" = "95b7a8237abc163d8ca31e49f7318efd" ] || die "owasp.zip md5 mismatch (Zenodo publishes 95b7a823…)"
  [ -d "$RAW/modsec/owasp" ] || ( cd "$RAW/modsec" && unzip -o -q owasp.zip -d owasp )
  have "$EXT/modsec.parquet" || $PY -m src.content_detector.ingest_modsec --src "$RAW/modsec/owasp" --out "$EXT/modsec.parquet"
  assert_rows "$EXT/modsec.parquet" 15262
}
fetch_ita(){ say "ita — NASA, ClarkNet and Calgary, from the Internet Traffic Archive"
  b="http://ita.ee.lbl.gov/traces"
  have "$EXT/ita/nasa.gz"    || dl "$b/NASA_access_log_Jul95.gz"    "$EXT/ita/nasa.gz"    || die "ita nasa"
  have "$EXT/ita/clark.gz"   || dl "$b/clarknet_access_log_Sep4.gz" "$EXT/ita/clark.gz"   || die "ita clarknet"
  have "$EXT/ita/calgary.gz" || dl "$b/calgary_access_log.gz"       "$EXT/ita/calgary.gz" || die "ita calgary"
  assert_loader nasa 8436; assert_loader clarknet 18411; assert_loader calgary 8152
}
fetch_csic(){ say "csic — Kaggle ispangler/csic-2010-web-application-attacks"
  kaggle_fetch ispangler/csic-2010-web-application-attacks "$RAW/csic" csic_database.csv
  assert_loader csic 25598 9644                                  # 9,644 benign / 15,954 attack after dedup
}
fetch_corpus(){ say "testbed corpus — frozen release asset holding data/corpus/labelled/testbed.parquet"
  tgz="$RAW/testbed_corpus.tar.gz"
  if [ -f "$tgz" ] && [ "$(sha256_of "$tgz")" = "$CORPUS_SHA256" ]; then echo "   [skip] $tgz present, checksum OK"
  else curl -fL "$CORPUS_RELEASE_URL" -o "$tgz" || die "corpus download ($CORPUS_RELEASE_URL)"; fi
  got=$(sha256_of "$tgz"); [ "$got" = "$CORPUS_SHA256" ] || die "corpus tarball sha256 mismatch: got $got"
  tar -xzf "$tgz" -C .                                            # writes data/corpus/labelled/testbed.parquet only
  assert_rows "$LAB/testbed.parquet" 21136 10908                  # 10,908 benign / 10,228 attack
}

ALL="zanbil srbh edgar weblog modsec ita csic corpus"
[ $# -eq 0 ] && set -- $ALL
for d in "$@"; do
  case "$d" in
    zanbil) fetch_zanbil ;; srbh) fetch_srbh ;; edgar) fetch_edgar ;; weblog) fetch_weblog ;;
    modsec) fetch_modsec ;; ita) fetch_ita ;; csic) fetch_csic ;; corpus) fetch_corpus ;;
    *) die "unknown dataset: $d (choices: $ALL)" ;;
  esac
done
say "fetch_data.sh done — parquets under $EXT, corpus under $LAB"
