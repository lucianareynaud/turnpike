# Security Policy

## Supported Versions

Security-related fixes, if any, may be applied only to the latest published version.

## Reporting a Vulnerability

Please do not open public issues for suspected security vulnerabilities.

Use [GitHub Security Advisories](https://github.com/lucianareynaud/turnpike/security/advisories/new)
to report vulnerabilities privately.

## Security Notes

Turnpike is a library, not a hosted service. Its security posture depends heavily on the way it is integrated and deployed.

Operational assumptions:
- Secrets such as provider API keys and OTLP credentials should be managed by the host environment.
- Provider SDKs and dependencies should be installed from trusted sources and pinned in a lockfile.
- Telemetry backends and OTLP collector endpoints should be treated as trusted infrastructure.

Operational cautions:
- Prompts, completions, and metadata may contain sensitive or untrusted content.
- Review what is emitted into logs, spans, and telemetry sinks before using the library in production.

## Recommendations

For production deployments:
- store credentials in a secret manager rather than committed files
- restrict network egress to authorised telemetry endpoints
- pin dependencies and run dependency vulnerability checks in CI