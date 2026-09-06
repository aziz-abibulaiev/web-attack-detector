"""Runs an attack tool as a subprocess, times the window, and writes a campaign manifest with
the family, tool, source IP, window, invocation and the tail of the tool's output. The proxy
captures the flows; this wrapper only records provenance. Used for ffuf, nuclei, hydra and
sqlmap.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.manifest import now_iso, write_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--tool", required=True)
    ap.add_argument("--app", required=True)
    ap.add_argument("--src-ip", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--label", default="attack")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command...>")
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("no command given", file=sys.stderr)
        return 2

    start_ts = now_iso()
    print(f"[run_tool] {args.campaign_id} ({args.tool}/{args.family}): {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    end_ts = now_iso()
    tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-500:]
    print(tail[-1200:], flush=True)

    write_manifest(
        out_dir=args.manifest_dir,
        campaign_id=args.campaign_id,
        app=args.app,
        label=args.label,
        attack_family=args.family if args.label == "attack" else None,
        attack_tool=args.tool if args.label == "attack" else None,
        src_ip=args.src_ip,
        start_ts=start_ts,
        end_ts=end_ts,
        extra={"invocation": cmd, "returncode": proc.returncode},
    )
    print(f"[run_tool] done {args.campaign_id} rc={proc.returncode} window [{start_ts},{end_ts}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
