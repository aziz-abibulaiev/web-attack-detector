"""Scripted legitimate traffic against a real running crAPI. Several simulated users sign up,
log in, and exercise the dashboard, vehicles, community with reads, posts and comments, shop
with browsing and orders, a single legitimate coupon validation, and mechanic requests: the
long tail of normal behavior where false positives live. Seeded, so only the pacing varies.
Writes only a benign campaign manifest.
"""

import argparse
import os
import random
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.manifest import now_iso, write_manifest  # noqa: E402
from common.uas import REALISTIC_UAS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--src-ip", required=True)
    ap.add_argument("--n-users", type=int, default=22)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-pace", type=float, default=0.05)
    ap.add_argument("--max-pace", type=float, default=0.35)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = args.target.rstrip("/")
    start_ts = now_iso()
    n = 0
    features = set()

    def pace():
        time.sleep(rng.uniform(args.min_pace, args.max_pace))

    def call(method, path, tok=None, tag=None, **kw):
        nonlocal n
        h = kw.pop("headers", {})
        h["User-Agent"] = rng.choice(REALISTIC_UAS)
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        try:
            r = requests.request(method, base + path, headers=h, timeout=15, **kw)
            n += 1
            if tag:
                features.add(tag)
            return r
        except requests.RequestException:
            return None
        finally:
            pace()

    run = args.campaign_id.split("_")[-1]
    for i in range(args.n_users):
        email = f"corpususer_{run}_{i}@example.com"
        pw = f"Crapi!{rng.randint(10000,99999)}"
        call("POST", "/identity/api/auth/signup", tag="signup",
             json={"name": f"Corpus User {i}", "email": email,
                   "number": f"415{rng.randint(1000000,9999999)}", "password": pw})
        r = call("POST", "/identity/api/auth/login", tag="login",
                 json={"email": email, "password": pw})
        tok = None
        if r is not None and r.status_code == 200:
            try:
                tok = r.json().get("token")
            except Exception:
                tok = None
        if not tok:
            continue
        call("GET", "/identity/api/v2/user/dashboard", tok, "dashboard")
        call("GET", "/identity/api/v2/vehicle/vehicles", tok, "vehicle")
        call("GET", "/workshop/api/shop/products", tok, "shop_browse")
        call("GET", "/community/api/v2/community/posts/recent", tok, "community_read")
        if rng.random() < 0.7:
            call("POST", "/community/api/v2/community/posts", tok, "community_post",
                 json={"title": f"My {rng.choice(['trip','review','question'])} {i}",
                       "content": f"Post content from user {i}: {rng.choice(['great','ok','nice'])}."})
        if rng.random() < 0.5:
            call("POST", "/community/api/v2/coupon/validate-coupon", tok, "coupon",
                 json={"coupon_code": rng.choice(["TRAC075", "TRAC15"])})
        if rng.random() < 0.6:
            call("GET", "/workshop/api/mechanic/", tok, "mechanic")
        if rng.random() < 0.4:
            call("POST", "/workshop/api/shop/orders", tok, "order",
                 json={"product_id": rng.choice([1, 2]), "quantity": rng.randint(1, 2)})

    end_ts = now_iso()
    write_manifest(
        out_dir=args.manifest_dir, campaign_id=args.campaign_id, app="crapi",
        label="benign", src_ip=args.src_ip, start_ts=start_ts, end_ts=end_ts,
        extra={"n_users": args.n_users, "seed": args.seed, "flows_sent": n,
               "feature_coverage": sorted(features)},
    )
    print(f"[crapi-benign] {n} flows, {args.n_users} users, features={sorted(features)}, "
          f"window [{start_ts},{end_ts}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
