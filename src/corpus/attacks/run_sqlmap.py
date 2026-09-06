"""Runs sqlmap against VAmPI's SQL-injectable endpoint GET /users/v1/{username} through the
capture proxy, so every probe lands in the unified schema. Writes an attack campaign manifest
naming the family, tool, source IP and window, which is the only basis for labeling these
flows: the labeller never sees a payload, it matches source IP and timestamp into the window.

sqlmap's internal request ordering is not fully deterministic. The version is pinned and the
exact invocation is recorded in the manifest, so the set of requests is reproducible even
when their order is not.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.manifest import now_iso, write_manifest  # noqa: E402


def sqlmap_version() -> str:
    try:
        out = subprocess.run(["sqlmap", "--version"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as e:
        return f"unknown ({e})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="proxy base URL, e.g. http://172.30.0.3:8080")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--src-ip", required=True)
    ap.add_argument("--endpoint", default="/users/v1/admin", help="injectable path to fuzz")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--risk", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = args.target.rstrip("/")
    url = base + args.endpoint
    ver = sqlmap_version()
    start_ts = now_iso()

    # --fresh-queries and a private output dir keep runs independent and reproducible.
    out_dir = f"/tmp/sqlmap_{args.campaign_id}"
    cmd = [
        "sqlmap", "-u", url,
        "--batch",                    # never prompt
        f"--level={args.level}",
        f"--risk={args.risk}",
        "--fresh-queries",
        "--flush-session",
        "--technique=BEUST",          # boolean, error, union, stacked, time
        "--random-agent",
        f"--output-dir={out_dir}",
        "--disable-coloring",
    ]
    print(f"[sqlmap] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    end_ts = now_iso()
    print(proc.stdout[-2000:], flush=True)
    if proc.returncode != 0:
        print(f"[sqlmap] stderr tail:\n{proc.stderr[-1000:]}", flush=True)

    write_manifest(
        out_dir=args.manifest_dir,
        campaign_id=args.campaign_id,
        app="vampi",
        label="attack",
        attack_family="sqli",
        attack_tool="sqlmap",
        src_ip=args.src_ip,
        start_ts=start_ts,
        end_ts=end_ts,
        extra={
            "tool_version": ver,
            "invocation": cmd,
            "endpoint": args.endpoint,
            "level": args.level,
            "risk": args.risk,
            "returncode": proc.returncode,
        },
    )
    print(f"[sqlmap] done rc={proc.returncode} window [{start_ts}, {end_ts}] ({ver})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
