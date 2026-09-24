# AIR Web 3.2 Public UI

AIR Web 3.2 is a post-release browser-surface improvement for AIR 0.9.12.

It does not change AIR inference semantics, scheduling, model ownership, Decision scoring, resource ownership, or backend execution.

Changes from Web 3.1:

- removes developer-machine paths from the public Setup view;
- adds public-safe clone/build/test/run guidance;
- adds an About / Architecture view for the R&D project;
- adds Home quick actions;
- labels AIR 0.9.12 explicitly as a public R&D checkpoint;
- keeps the hardened non-streaming default and SSE parser from Web 3.1;
- slows background polling slightly to reduce unnecessary local telemetry traffic;
- preserves the rule that the browser is a surface, not a runtime state owner.

The frozen AIR 0.9.12 release artifact remains authoritative for release qualification. Web 3.2 is intended for the public repository `main` branch after the v0.9.12 release tag.
