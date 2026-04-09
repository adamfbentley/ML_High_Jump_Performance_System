---
mode: agent
description: Add a new public biomechanics dataset loader that produces BiomechanicalSample objects
tools:
  - read_file
  - file_search
  - grep_search
  - create_file
  - replace_string_in_file
  - run_in_terminal
---

Add a new dataset loader for the dataset described below.

## Requirements

1. Read `src/data_pipeline/sample.py` first — the loader **must** produce valid `BiomechanicalSample` objects
2. Read `src/data_pipeline/loaders/opencap.py` as a reference implementation
3. Read `src/data_pipeline/registry.py` and add the new dataset to `DATASET_REGISTRY`
4. Place the loader at `src/data_pipeline/loaders/<dataset_name>.py`
5. Export it from `src/data_pipeline/loaders/__init__.py`
6. Write at least 2 pytest tests in `tests/test_data_pipeline/test_loaders.py` using temp files
7. Run `pytest tests/ --ignore=tests/test_pinn` to confirm all tests pass

## Physics constraints

- Joint angles must be converted **deg → rad** on ingestion using `np.deg2rad()`
- GRF must be in **Newtons** (check and convert if dataset uses bodyweights or kg-force)
- CoM position must be in **metres** (convert from mm if necessary)
- Time axis must be in **seconds**

## Dataset to add

<!-- Describe the dataset here: file format, column names, directory structure, download URL, licence -->
