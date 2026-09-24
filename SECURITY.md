# Security

AIR is research software. Treat the local HTTP service as a privileged local
development/runtime interface.

## Safe default deployment

- Bind AIR to loopback (`127.0.0.1`) unless you have deliberately designed a
  secured network deployment.
- Do not expose an unauthenticated AIR server directly to the public internet.
- Do not run AIR with more operating-system privilege than it needs.
- Treat model files as untrusted input until validated.
- Keep GPU drivers, compiler toolchains, and system libraries patched.
- Do not paste secrets into prompts or diagnostics that you intend to publish.

The browser command center does not require arbitrary shell execution and
should never be extended with a generic command endpoint.

A future setup/bootstrap helper must remain loopback-only by default, use
allowlisted operations, and require explicit confirmation for privileged
changes.

## Reporting a vulnerability

Please avoid publishing exploit details in a normal issue before the maintainer
has had a chance to review them. If GitHub private vulnerability reporting is
enabled for the repository, use that mechanism. Otherwise contact the
repository owner privately through the contact method listed on their GitHub
profile and include enough information to reproduce the issue safely.
