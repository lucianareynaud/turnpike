# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do not open public issues for security vulnerabilities.**

Use [GitHub Security Advisories](https://github.com/lucianareynaud/turnpike/security/advisories/new)
to report vulnerabilities privately. You will receive an acknowledgement
within 48 hours and a detailed response within 7 business days.

## Trust Model

### What Turnpike trusts

- **Environment variables** — configuration values (`OTEL_*`, provider API keys)
  are assumed to come from a secure runtime (e.g. a secret manager or CI vault).
- **Provider SDK integrity** — the OpenAI, Anthropic, and other provider SDKs
  are assumed to be unmodified packages installed from PyPI.
- **OTel exporter endpoints** — the configured OTLP collector endpoint is assumed
  to be a trusted, authenticated receiver.

### What Turnpike does NOT trust

- **User prompt content** — prompts and completions may contain arbitrary text
  and are treated as untrusted data in all instrumentation paths.
- **Metadata values** — model names, tag values, and other user-supplied metadata
  are validated and sanitised before being emitted as span attributes.

## Recommendations for Production Deployments

- Store API keys and OTLP credentials in a dedicated secret manager
  (e.g. AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager)
  rather than plain environment variables or `.env` files.
- Restrict network egress so that OTel exporters can only reach your
  authorised collector endpoints.
- Pin all dependencies (including transitive ones) with a lockfile and
  run `pip-audit` in CI to catch known vulnerabilities.
