---
mode: agent
description: Run the full test suite and fix any failures
tools:
  - run_in_terminal
  - read_file
  - replace_string_in_file
  - grep_search
---

Run the test suite and fix all failures.

## Run tests

```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ --ignore=tests/test_pinn -v --tb=short 2>&1
```

To include PINN tests (requires torch, slower):
```powershell
& ".venv\Scripts\python.exe" -m pytest tests/ -v --tb=short 2>&1
```

## Fix failures

For each failing test:
1. Read the test to understand what it expects
2. Read the source function being tested
3. Fix the source (not the test) unless the test itself has a bug
4. Re-run only the failing test to confirm the fix: `pytest tests/path/to/test.py::test_name`

## Physics tests — special handling

Tests named `test_*_conserv*`, `test_*_fma*`, or `test_*_residual*` are physics-law
verification tests. **Never relax their tolerances** — if they fail, the physics
implementation is wrong and must be fixed.

For numerical tolerance in other tests: use `atol=1e-3` as a starting point.
Only relax further if edge effects (e.g. `np.gradient` at array boundaries) are
the cause, and add a comment explaining why.

## After fixing

Run the full suite one final time and confirm: `N passed, 0 failed`
