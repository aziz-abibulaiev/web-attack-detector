#!/usr/bin/env bash
# Smoke-run entrypoint: VAmPI + mitmproxy + one benign campaign + one sqlmap campaign,
# then provenance labeling and verification. Reproduces the whole small run end to end.
#
# Design that makes provenance clean:
#   - the benign generator runs from 172.30.0.10 and sqlmap from 172.30.0.20, distinct IPs;
#   - the two campaigns run in non-overlapping windows: benign finishes, the DB is reset, and
#     only then does the attack start;
#   - the mitmproxy addon records the client peer IP the proxy sees, which equals the
#     generator's static IP, and the labeller matches that IP and the timestamp into a window.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS="$REPO/src/corpus"
DATA="$REPO/data/corpus_smoke"
NET="corpus_net"
PROXY_IP="172.30.0.3"
BENIGN_IP="172.30.0.10"
ATTACK_IP="172.30.0.20"
export RUN_ID="smoke_$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$DATA/raw" "$DATA/manifests" "$DATA/labelled"
: > "$DATA/raw/vampi_smoke.jsonl"   # fresh capture file for this run

echo "== [1/7] build generator image =="
docker build -q -f "$CORPUS/Dockerfile.generator" -t corpus-generator "$CORPUS" >/dev/null

echo "== [2/7] bring up VAmPI + capture proxy =="
docker compose -f "$CORPUS/docker-compose.smoke.yaml" up -d

run_gen() {  # role_ip  script_and_args...
  local ip="$1"; shift
  docker run --rm --network "$NET" --ip "$ip" \
    -v "$CORPUS":/corpus:ro -v "$DATA":/data \
    corpus-generator "$@"
}

# Readiness and reset run from an infra IP that talks to VAmPI directly on 172.30.0.2:5000
# rather than through the proxy, so these admin requests are never captured and the capture
# file holds only the benign flows from .10 and the attack flows from .20. A bare TCP connect
# confirms the proxy is listening without emitting an HTTP flow.
echo "   waiting for VAmPI + proxy..."
run_gen 172.30.0.5 python -c "
import socket, time, urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen('http://172.30.0.2:5000/', timeout=3); break
    except Exception: time.sleep(1)
else: raise SystemExit('vampi not ready')
for _ in range(60):
    try:
        socket.create_connection(('172.30.0.3',8080),3).close(); break
    except Exception: time.sleep(1)
else: raise SystemExit('proxy not ready')
print('vampi+proxy ready')
"

echo "== [3/7] benign campaign (src=$BENIGN_IP) =="
run_gen "$BENIGN_IP" python /corpus/benign/vampi_driver.py \
  --target "http://$PROXY_IP:8080" --manifest-dir /data/manifests \
  --campaign-id benign_vampi_01 --src-ip "$BENIGN_IP" --n-users 8 --seed 42

echo "== [4/7] (no mid-run reset: benign driver already seeded the DB; sqlmap only probes) =="

echo "== [5/7] sqlmap campaign (src=$ATTACK_IP) =="
run_gen "$ATTACK_IP" python /corpus/attacks/run_sqlmap.py \
  --target "http://$PROXY_IP:8080" --manifest-dir /data/manifests \
  --campaign-id sqli_vampi_sqlmap_01 --src-ip "$ATTACK_IP" \
  --endpoint /users/v1/admin --level 3 --risk 2 --seed 42

echo "== [6/7] tear down proxy (flush capture) =="
docker compose -f "$CORPUS/docker-compose.smoke.yaml" down >/dev/null 2>&1

echo "== [7/7] provenance labeling =="
"${PYTHON:-python3.13}" "$CORPUS/label/labeller.py" \
  --capture "$DATA/raw/vampi_smoke.jsonl" \
  --manifest-dir "$DATA/manifests" \
  --out "$DATA/labelled/vampi_smoke.parquet"

echo "== verification =="
"${PYTHON:-python3.13}" "$CORPUS/verify_smoke.py" \
  --parquet "$DATA/labelled/vampi_smoke.parquet" \
  --manifest-dir "$DATA/manifests" \
  --labeller "$CORPUS/label/labeller.py" \
  --run-id "$RUN_ID" \
  --manifest-out "$REPO/data/corpus_smoke/run_manifest.json"

echo "Smoke run complete. Labeled parquet: $DATA/labelled/vampi_smoke.parquet"
