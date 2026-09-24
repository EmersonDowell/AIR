# AIR Strategy Transition Model

AIR Strategy Lab must optimize state transitions, not only steady-state throughput.

## Feasibility first

A candidate tactic is infeasible if any hard constraint fails:

- correctness class is insufficient for the requested surface;
- backend capability does not advertise the operation tactic;
- prospective prepared artifact bytes exceed the device budget;
- request/KV capacity cannot be admitted after prospective preparation;
- required artifact preparation cannot be completed safely while the backend owns active sequences.

No score can override infeasibility.

## Conservative transition cost

For a currently resident tactic A and candidate B, model:

- P = preparation latency for B not already resident;
- E = eviction/transition latency required to make B resident;
- X = any other measured transition cost;
- R_A = current steady-state rate;
- R_B,L = lower confidence bound on candidate steady-state rate.

Transition cost:

`C_transition = P + E + X`

Conservative break-even work units:

`H_break = C_transition / (1000/R_A - 1000/R_B,L)`

when `R_B,L > R_A`; otherwise break-even is infinite.

AIR should switch only when the expected remaining workload horizon exceeds the conservative break-even horizon and the resource constraints remain feasible.

The research implementation in `equivalence_contract.hpp` encodes this pure decision model for property tests. It is not yet wired into the production planner.

## Why this matters

A tactic that is 2x faster after a 2-second preparation may be a poor choice for a 20-token interaction and an excellent choice for a long batch. Likewise, evicting a prepared artifact to meet a short-lived low-VRAM request can be harmful if a large high-throughput workload is expected immediately afterward.

The optimizer therefore eventually needs evidence for:

- preparation bytes and time;
- current prepared artifact residency;
- candidate steady-state confidence interval;
- expected workload horizon;
- explicit VRAM objective/budget;
- eviction cost;
- reuse probability or expected future horizon;
- a hysteresis/dwell policy that prevents tactic thrashing.

## Artifact identity

Prepared artifacts should be keyed by at least:

- canonical model digest;
- hardware/device fingerprint;
- tactic kind and tactic ABI/version;
- operation scope;
- packing/precision geometry.

A manifest must never claim a low-memory plan while an unrelated high-memory optional artifact remains invisibly resident.
