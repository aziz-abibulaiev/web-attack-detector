"""The unified corpus schema, one row per HTTP transaction.

Shared by the mitmproxy capture addon, which writes raw unlabeled rows, the provenance
labeller, and the splitter. The capture addon fills everything observable on the wire plus
the capture environment, and the labeller fills the provenance fields. Keeping the field list
in one place guarantees the parquet matches what every downstream reader expects.

Nothing in this module inspects payload content to decide a label. label, attack_family,
attack_tool and label_source come only from campaign manifests, and session_id comes only
from the JWT sub claim or the source IP fallback.
"""

# Full unified schema, in emit order. Types are documented in docs/ and enforced in parquet.
UNIFIED_FIELDS = [
    "capture_id",            # str  uuid5 of run_id and seq, deterministic per run and seq
    "timestamp",             # str  ISO-8601 with ms, UTC
    "src_ip",                # str
    "src_port",              # int
    "http_method",           # str
    "url_path",              # str  decoded once for readability
    "url_path_raw",          # str  exactly as on the wire
    "url_query",             # str  raw query string
    "http_version",          # str
    "host",                  # str
    "headers",               # str  JSON object of request headers actually present
    "request_body",          # str  raw body, "" if none
    "request_content_type",  # str
    "response_status",       # int  nullable if the app never answered
    "response_size",         # int
    "response_time_ms",      # float
    "app",                   # str  crapi | vampi | zanbil | srbh
    "session_id",            # str  behavioral grouping key
    "label",                 # str  benign | attack
    "attack_family",         # str  null for benign
    "attack_tool",           # str  null for benign
    "campaign_id",           # str  which generation run produced this row
    "label_source",          # str  provenance string
]

# Fields the capture addon can populate from the wire + capture environment. The remaining
# fields session_id, label, attack_family, attack_tool, campaign_id, label_source are the
# labeller's job and must not be set at capture time.
CAPTURE_FIELDS = [
    "capture_id", "timestamp", "src_ip", "src_port", "http_method",
    "url_path", "url_path_raw", "url_query", "http_version", "host",
    "headers", "request_body", "request_content_type",
    "response_status", "response_size", "response_time_ms", "app",
]

PROVENANCE_FIELDS = [
    "session_id", "label", "attack_family", "attack_tool", "campaign_id", "label_source",
]

# Controlled vocabulary for attack_family mapped to OWASP in the paper's coverage table.
ATTACK_FAMILIES = [
    "sqli", "xss", "cmd_injection", "nosql_injection", "ldap_injection", "ssti",
    "path_traversal", "ssrf", "scan", "brute_force", "cred_stuffing",
    "resource_exhaustion", "bola_idor", "mass_assignment", "business_logic_abuse",
]

ATTACK_TOOLS = ["sqlmap", "zap", "nuclei", "ffuf", "wfuzz", "hydra", "patator", "custom"]

APPS = ["crapi", "vampi", "zanbil", "srbh"]

# Parquet dtypes. response_status and response_size stay nullable Int64, because a request
# that never reached the app has no response.
PARQUET_DTYPES = {
    "capture_id": "string", "timestamp": "string", "src_ip": "string",
    "src_port": "Int64", "http_method": "string", "url_path": "string",
    "url_path_raw": "string", "url_query": "string", "http_version": "string",
    "host": "string", "headers": "string", "request_body": "string",
    "request_content_type": "string", "response_status": "Int64",
    "response_size": "Int64", "response_time_ms": "float64", "app": "string",
    "session_id": "string", "label": "string", "attack_family": "string",
    "attack_tool": "string", "campaign_id": "string", "label_source": "string",
}
