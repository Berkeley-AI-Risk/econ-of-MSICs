# Validation

Run the commands in the [README](../README.md) from the repository root.

| Check | What it verifies |
| --- | --- |
| `test_msic_analysis.py` | Figure/app agreement, break-even behavior, exhaustive cost minimization on a toy whitelist, parameter sweeps, scale invariance, pricing calculations, historical shifts, app controls, and a stress case independent of the customizable figure grid |
| `test_reproduction.py` | Selection of the manifest's exact captures, rejection of an incorrect checksum, catalog and historical-data provenance, and daily-collection date windows with network calls mocked |
| `sm_asic_topn/static/verify_outputs.py` | Independent raw-data reconstruction of top-N coverage and set-overlap results, service-variant aggregation, CSV primary keys, and input hashes |
| `openrouter_demand.py verify` | Processed-file presence, nonempty and nonduplicate rows, and historical-archive cross-validation |
| `scripts/check_reproduction.py` | Reconstruction of committed derived CSVs from pinned captures in temporary space |

The independent static verifier does not import the data generator.
Its numerical comparisons use an absolute tolerance of `2e-12`.
The full rebuild compares nonnumeric CSV fields exactly and allows absolute
and relative tolerances of `1e-12` for floating-point fields. It adds a newer
decoy capture to check that the generator still uses the pinned manifest.

The [GitHub Actions workflow](../.github/workflows/tests.yml) runs the tests,
data reconstruction, and figure generation on Python 3.12, then checks that
tracked files remain unchanged.
