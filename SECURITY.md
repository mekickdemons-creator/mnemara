# Security Policy

## Threat model

Mnemara is a **runtime that gives an LLM tool access to the machine it
runs on** (Bash, Read, Write, Edit, MCP). It is not a sandbox. Within
its design, the operator's permission settings are the security
boundary — see the [Permissions model](README.md#permissions-model)
section of the README.

For the purpose of this policy, a security issue is one of:

- A way to escape Mnemara's documented permission model (e.g. running
  Bash when Bash is set to `deny`, writing outside `$HOME` when
  `file_tool_home_only` is `True`, escaping the instance directory).
- A way for a malicious **role doc, MCP server, or stored turn** to
  cause Mnemara to do something the operator did not authorize.
- A way for a network adversary (during normal API traffic) to
  compromise the local machine.
- An issue in a transitive dependency that Mnemara's usage exposes in
  a non-obvious way.

A user-authorized `rm -rf` from an LLM with Bash=allow is **not** a
security issue — it's the documented behavior of permissive settings.

## Reporting a vulnerability

**Please do not file public GitHub issues for security problems.**

Use one of these private channels:

- **GitHub Security Advisories** — preferred. Open a draft advisory at
  <https://github.com/mekickdemons-creator/mnemara/security/advisories/new>.
- **Email** — `mekickdemons@gmail.com` with subject prefix `[mnemara
  security]`.

Include:

1. Mnemara version (`mnemara --version` or the installed PyPI version).
2. Python version, OS, and (if relevant) the role doc + permission
   settings in use.
3. A minimal reproduction or proof of concept.
4. Your assessment of severity and any suggested mitigation.

## What to expect

- Acknowledgement within **3 business days**.
- An initial assessment (confirm / dispute / need more info) within
  **7 business days**.
- For confirmed issues: a fix released to PyPI and a published GitHub
  Security Advisory crediting the reporter (unless they request
  otherwise).
- Coordinated disclosure: please give us at least **30 days** before
  public disclosure, longer for issues that require dependency
  coordination.

This is a single-maintainer project. Response times reflect that —
they are best-effort, not contractual.

## Out of scope

- Misuse of permissive settings (Bash=allow, broad regex in
  `permissions.json`, `file_tool_home_only=False`). These are
  documented foot-guns; the README warns about them.
- Issues in `claude-agent-sdk`, the Anthropic API, or other upstream
  dependencies — please report those to their respective projects.
  We will, however, ship a Mnemara-side mitigation when one exists.
- Prompt-injection of the operator's own conversation. The operator is
  trusted relative to the LLM; if your role doc or memory contents tell
  the agent to do something dangerous, that is an operator decision,
  not a Mnemara vulnerability.
