"""Regexes the evaluation drivers use to explain and inspect results: the decoded-marker check
on top-ranked alerts, the regex recall baseline, and the per-family cross-check in family_eval.
They are display and measurement only, never a model feature, a label or a threshold.
"""

from __future__ import annotations

import re

# ---- post-hoc explanation rules: display only, never features and never labels ----
_EXPLAIN_RULES = {
    "sqli": re.compile(r"(union\s+select|\bor\s+1=1\b|'\s*or\s*'1'='1|information_schema|sleep\s*\(|waitfor\s+delay|--\s|;\s*drop\s+table)", re.I),
    "xss": re.compile(r"(<script|onerror\s*=|onload\s*=|javascript:|<svg|<img[^>]+src|document\.cookie|alert\s*\()", re.I),
    "path_traversal": re.compile(r"(\.\./|\.\.\\|%2e%2e%2f|/etc/passwd|/etc/shadow|win\.ini|boot\.ini|php://)", re.I),
    "cmd_injection": re.compile(r"(;\s*(cat|ls|id|whoami|uname|nc|curl|wget)|\|\s*(id|sh|bash)|\$\(|`[^`]+`|&&\s*\w+)", re.I),
    "ssti": re.compile(r"(\{\{.*?\}\}|\$\{.*?\}|<%=|#\{.*?\}|\{%.*?%\})", re.I),
    "ssrf": re.compile(r"(169\.254\.169\.254|localhost|127\.0\.0\.1|metadata\.google|gopher://|dict://|file://|0\.0\.0\.0)", re.I),
    "nosql_injection": re.compile(r"(\$ne\b|\$gt\b|\$where\b|\$regex\b|\$in\b|\$exists\b)", re.I),
    "scan": re.compile(r"(/\.git|/\.env|/wp-admin|/phpinfo|/\.htpasswd|/actuator|/swagger|/\.aws)", re.I),
}
