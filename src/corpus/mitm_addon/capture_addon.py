"""A mitmproxy addon that captures every flow to raw JSONL in the unified capture schema.

Runs in reverse-proxy mode in front of one app and records what is on the wire plus the
capture environment: app name, client peer IP from the docker network, and proxy timing. It
writes one JSON object per flow, appended and flushed immediately, so a crash mid-run loses
at most the in-flight request.

It does not assign label, attack_family, campaign_id or session_id. Those come later from
corpus/label/labeller.py and the campaign manifests. That separation is what keeps labels
independent of content: this file never looks at a payload to decide anything, and the
labeller never looks at a payload at all.

Options, all required, set with --set on the mitmdump command line:
    capture_out   path to the JSONL file to append to
    capture_app   app name recorded in every row, for example vampi
    capture_run   run_id, used to make capture_id deterministic

mitmproxy timestamps are epoch seconds as floats, and response_time_ms is
response.timestamp_end minus request.timestamp_start, times 1000.
"""

import json
import threading
import uuid
from datetime import datetime, timezone

from mitmproxy import ctx, http

_UUID_NS = uuid.UUID("6f1a7b2c-0000-4000-8000-000000000004")  # fixed namespace for this project


class CorpusCapture:
    def __init__(self) -> None:
        self._fh = None
        self._seq = 0
        self._lock = threading.Lock()

    def load(self, loader) -> None:
        loader.add_option("capture_out", str, "", "JSONL output path")
        loader.add_option("capture_app", str, "", "app name recorded per row")
        loader.add_option("capture_run", str, "", "run_id for deterministic capture_id")

    def running(self) -> None:
        if not ctx.options.capture_out:
            raise ValueError("capture_out option is required")
        # line-buffered append; each flow is flushed so nothing is lost on kill
        self._fh = open(ctx.options.capture_out, "a", buffering=1, encoding="utf-8")
        ctx.log.info(f"[corpus-capture] writing to {ctx.options.capture_out} app={ctx.options.capture_app}")

    def done(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()

    @staticmethod
    def _iso(ts: float | None) -> str:
        t = ts if ts else datetime.now(timezone.utc).timestamp()
        return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
            f"{int((t % 1) * 1000):03d}Z"

    def response(self, flow: http.HTTPFlow) -> None:
        self._emit(flow, reached_app=True)

    def error(self, flow: http.HTTPFlow) -> None:
        # the request left the client but the app never answered: connection error or timeout.
        # Recorded so that requests which never reached the app can be excluded.
        if flow.response is None:
            self._emit(flow, reached_app=False)

    def _emit(self, flow: http.HTTPFlow, reached_app: bool) -> None:
        with self._lock:
            seq = self._seq
            self._seq += 1

        req = flow.request
        peer = flow.client_conn.peername or ("", 0)
        raw_path = req.path or "/"
        if "?" in raw_path:
            url_path_raw, url_query = raw_path.split("?", 1)
        else:
            url_path_raw, url_query = raw_path, ""

        try:
            body_bytes = req.raw_content or b""
            body = body_bytes.decode("utf-8", errors="replace")
        except Exception:
            body = ""

        resp = flow.response
        if resp is not None:
            status = int(resp.status_code)
            size = len(resp.raw_content or b"")
            t_end = resp.timestamp_end or req.timestamp_start
        else:
            status = None
            size = 0
            t_end = req.timestamp_end or req.timestamp_start

        rt_ms = None
        if req.timestamp_start and t_end:
            rt_ms = round((t_end - req.timestamp_start) * 1000.0, 3)

        # decode the path once for readability; keep raw untouched
        try:
            from urllib.parse import unquote
            url_path = unquote(url_path_raw)
        except Exception:
            url_path = url_path_raw

        row = {
            "capture_id": str(uuid.uuid5(_UUID_NS, f"{ctx.options.capture_run}:{seq}")),
            "timestamp": self._iso(req.timestamp_start),
            "src_ip": str(peer[0]),
            "src_port": int(peer[1]) if len(peer) > 1 else 0,
            "http_method": req.method,
            "url_path": url_path,
            "url_path_raw": url_path_raw,
            "url_query": url_query,
            "http_version": req.http_version,
            "host": req.host_header or req.pretty_host or "",
            "headers": json.dumps(dict(req.headers), ensure_ascii=False),
            "request_body": body,
            "request_content_type": req.headers.get("content-type", ""),
            "response_status": status,
            "response_size": size,
            "response_time_ms": rt_ms,
            "app": ctx.options.capture_app,
            "reached_app": reached_app,
        }
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")


addons = [CorpusCapture()]
