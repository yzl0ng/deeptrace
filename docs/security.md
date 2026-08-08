# Security and public-release boundary

## Trust boundaries

- user queries are untrusted input;
- model output is an untrusted proposal until parsed and validated;
- search results and web pages are untrusted evidence;
- only the Runtime owns canonical state, budgets and Evidence IDs;
- secrets remain server-side and must never enter browser props or SSE payloads.

## Repository policy

The public repository intentionally excludes:

- `.env` files and provider credentials;
- SSH keys, hosts and user-specific absolute paths;
- Cloud deployment resource IDs;
- SQLite/D1 exports containing user queries;
- raw provider traces that may contain private input;
- model weights, LoRA adapters and checkpoints;
- build caches and large experimental downloads.

Configuration templates use placeholders and environment variables. If a secret was ever committed, rotate it first and then remove it from Git history; a later deletion commit is insufficient.

## Public deployment checklist

Before exposing either runtime to the internet:

1. require authentication or a strong device/server access gate;
2. rate-limit by identity and source, not only by browser UI;
3. cap actions, tokens, searches, page reads, wall time and provider cost;
4. restrict outbound URL schemes, redirects, private network ranges and response sizes;
5. sanitize logs and define retention/deletion for queries and traces;
6. use prepared SQL statements and least-privilege bindings;
7. keep API keys in the hosting secret manager;
8. monitor provider errors, malformed actions, gate rejection and cost;
9. return generic public errors while retaining a private correlation ID;
10. test prompt-injection pages and unknown Evidence ID attacks.

## Reporting

Do not paste live credentials, private queries or infrastructure identifiers into a public GitHub issue. Revoke exposed credentials immediately before investigating the code path.
