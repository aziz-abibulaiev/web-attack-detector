"""Scripted legitimate traffic against a real running VAmPI; only the client behavior is
synthetic. Several simulated users register, log in, and browse and manage users and books,
at human-like pacing and in a realistic order. No attack tool touches the app while this
runs, which the orchestrator enforces with non-overlapping windows and a distinct source IP.

It writes no labels, only a benign campaign manifest with the window and source IP that the
labeller later uses. Everything derives from --seed and only wall-clock pacing varies, so the
set of requests is reproducible.

Endpoints, from VAmPI's OpenAPI:
    GET  /                          POST /users/v1/register
    GET  /createdb                  POST /users/v1/login   -> JWT, sub = username
    GET  /users/v1                  GET  /me               [auth]
    GET  /users/v1/{username}       PUT  /users/v1/{username}/email    [auth]
    GET  /books/v1                  PUT  /users/v1/{username}/password [auth]
    POST /books/v1 [auth]           GET  /books/v1/{book_title}        [auth]

    [auth] marks the endpoints sent with the Bearer token.
"""

import argparse
import os
import random
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.manifest import now_iso, write_manifest  # noqa: E402

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "PostmanRuntime/7.36.0",
    "python-requests/2.32.3",
    "VAmPI-mobile/1.4 (iOS 17.2)",
]

FIRST = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="proxy base URL, e.g. http://172.30.0.3:8080")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--src-ip", required=True, help="this container's static IP, for the manifest")
    ap.add_argument("--n-users", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-pace", type=float, default=0.02)
    ap.add_argument("--max-pace", type=float, default=0.2)
    ap.add_argument("--skip-createdb", action="store_true",
                    help="do not reset the DB (for parallel benign containers)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = args.target.rstrip("/")
    start_ts = now_iso()
    n_flows = 0

    def ua() -> str:
        return rng.choice(USER_AGENTS)

    def pace() -> None:
        time.sleep(rng.uniform(args.min_pace, args.max_pace))

    def req(method, path, session=None, **kw):
        nonlocal n_flows
        headers = kw.pop("headers", {})
        headers.setdefault("User-Agent", ua())
        s = session or requests
        try:
            r = s.request(method, base + path, headers=headers, timeout=15, **kw)
            n_flows += 1
            return r
        except requests.RequestException as e:
            print(f"[benign] {method} {path} -> error {e}", flush=True)
            return None
        finally:
            pace()

    # seed and reset the app database so the run is repeatable. Parallel workers skip it;
    # otherwise they would wipe each other's registered users mid-run.
    req("GET", "/")
    if not args.skip_createdb:
        req("GET", "/createdb")
    req("GET", "/users/v1")
    req("GET", "/books/v1")

    users = []
    for i in range(args.n_users):
        name = f"{rng.choice(FIRST)}{rng.randint(100, 999)}"
        pwd = f"Pw{rng.randint(10000, 99999)}!"
        email = f"{name}@example.com"
        users.append({"username": name, "password": pwd, "email": email})

    tokens = {}
    for u in users:
        # register then log in — a normal onboarding flow
        req("POST", "/users/v1/register",
            json={"username": u["username"], "password": u["password"], "email": u["email"]})
        r = req("POST", "/users/v1/login",
                json={"username": u["username"], "password": u["password"]})
        if r is not None and r.status_code == 200:
            try:
                tok = r.json().get("auth_token") or r.json().get("token")
                if tok:
                    tokens[u["username"]] = tok
            except Exception:
                pass

    def auth(username):
        t = tokens.get(username)
        return {"Authorization": f"Bearer {t}"} if t else {}

    # authenticated browsing / management, realistic order
    for u in users:
        h = auth(u["username"])
        if not h:
            continue
        req("GET", "/me", headers=h)
        req("GET", f"/users/v1/{u['username']}", headers=h)
        # each user adds a couple of books, then lists and reads them back
        for b in range(rng.randint(1, 3)):
            title = f"{u['username']}-book-{b}"
            req("POST", "/books/v1", headers=h, json={"book_title": title, "secret": f"s3cr3t-{b}"})
            req("GET", "/books/v1", headers=h)
            req("GET", f"/books/v1/{title}", headers=h)
        # occasional profile updates
        if rng.random() < 0.5:
            req("PUT", f"/users/v1/{u['username']}/email", headers=h,
                json={"email": f"{u['username']}.new@example.com"})
        if rng.random() < 0.3:
            newpw = f"Pw{rng.randint(10000, 99999)}!"
            req("PUT", f"/users/v1/{u['username']}/password", headers=h,
                json={"password": newpw})

    # some anonymous browsing sprinkled in
    for _ in range(rng.randint(6, 12)):
        who = rng.choice(users)["username"]
        req("GET", rng.choice([f"/users/v1/{who}", "/users/v1", "/books/v1", "/"]))

    end_ts = now_iso()
    write_manifest(
        out_dir=args.manifest_dir,
        campaign_id=args.campaign_id,
        app="vampi",
        label="benign",
        src_ip=args.src_ip,
        start_ts=start_ts,
        end_ts=end_ts,
        extra={"n_users": args.n_users, "seed": args.seed, "flows_sent": n_flows},
    )
    print(f"[benign] done: {n_flows} flows, window [{start_ts}, {end_ts}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
