# Contributing to AIR

AIR is an R&D project. Contributions are welcome, but architectural changes
need evidence rather than benchmark-only arguments.

## Before opening a pull request

1. Build AIR from a clean tree.
2. Run the full test suite.
3. Keep one production inference path and one scheduler architecture.
4. Do not add hidden mutable state or a second owner for model/runtime truth.
5. Preserve public API semantics unless the change explicitly proposes and
   documents a contract revision.
6. Include measurements for performance claims.
7. Include correctness or characterization tests before large refactors.

Basic verification:

```bash
./scripts/build.sh
ctest --test-dir build --output-on-failure
```

For web-only changes, also run the browser package checks where available and
verify that the UI still talks only to AIR's public HTTP contract.

## Research changes

For a performance or architecture change, describe:

- the hypothesis;
- what evidence would falsify it;
- baseline configuration;
- hardware/software environment;
- correctness criteria;
- resource impact;
- negative results.

A slower but clearer/correct implementation may be more valuable than a narrow
benchmark win.

## License of contributions

Unless explicitly agreed otherwise, contributions intentionally submitted for
inclusion in this public repository are accepted under the Apache License 2.0.
