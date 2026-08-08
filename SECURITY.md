# Security policy

Please do not report live API keys, private user queries, database exports, SSH
coordinates or other sensitive infrastructure data in a public issue.

For a suspected credential exposure:

1. revoke and rotate the credential immediately;
2. preserve only sanitized logs and the relevant commit ID;
3. contact the repository owner privately through the security contact listed
   on their GitHub profile;
4. remove the value from Git history before making a public remediation PR.

The project is a research implementation. Public deployments must add
authentication, server-side rate limits, cost ceilings, outbound URL controls,
trace retention limits and provider-secret management. See
[`docs/security.md`](docs/security.md) for the complete checklist.
