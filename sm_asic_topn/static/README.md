# Demand tables

[analyze_topn.py](analyze_topn.py) builds the demand tables from the raw
day, week, and catalog captures listed in
[analysis_manifest.json](analysis_manifest.json). Input hashes are checked
before loading.

## Selecting the paper's demand distribution

In `ranked_units_by_universe.csv`, select:

```text
period = week
universe = current_catalog_text_output
aggregation = model_permaslug
```

Use `model_permaslug` as the model identifier and
`token_share_within_universe` as its fractional demand share. This slice has
315 rows for the week ending August 11, 2026. The daily slice is separate.

`current_catalog_text_output` includes positive-demand identifiers that match
the captured catalog and have text among their output modalities.
`model_permaslug` sums service variants; `exact_model_variant` keeps them
separate. `all_observed` retains listings excluded by the catalog filter.

## Output files

| File | Contents |
| --- | --- |
| `ranked_units_by_universe.csv` | Identifiers, counts, ranks, and shares within each universe |
| `unit_rankings.csv` | All positive-demand units before universe filtering |
| `topn_coverage.csv` | Coverage and omitted shares for each feasible N and ranking measure |
| `selected_n_summary.csv` | Selected rows from the top-N coverage grid |
| `selection_stability.csv` | Set overlap and cross-period/cross-measure coverage |
| `design_count_sensitivity.csv` | Counts of ranking units, permaslugs, and attached variants |
| `metadata_availability.csv` | Availability of catalog fields and modality flags |
| `analysis_manifest.json` | Input paths, hashes, output row counts, and validation checks |

Coverage fields without `_of_all_observed` use the selected universe as their
denominator. Fields with that suffix use all observed traffic.
Prompt tokens, completion tokens, total tokens, and requests are distinct
measures. Catalog modality flags describe capability, not observed usage;
`hardware_compatible_weight_design_count=not_identified` is not a missing count
to fill with the number of service variants.

## Rebuild and verify

From the repository root:

```bash
python scripts/check_reproduction.py
python sm_asic_topn/static/analyze_topn.py verify
python sm_asic_topn/static/verify_outputs.py
```

The first command rebuilds in temporary space. To overwrite this directory's
derived CSVs using the pinned inputs, run:

```bash
python sm_asic_topn/static/analyze_topn.py analyze
```

The generator's `verify` command checks its computed tables without writing
files. `verify_outputs.py` independently checks the committed CSVs against the
raw captures.
