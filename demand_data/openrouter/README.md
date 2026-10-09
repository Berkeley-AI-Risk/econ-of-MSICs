# OpenRouter source data

## Pinned captures

The source captures date from August 12, 2026. Retrieval URLs, SHA-256 hashes,
and the data extractions are recorded in
[retrieval_manifest_20260812T181706Z.json](raw/retrieval_manifest_20260812T181706Z.json).

| File in `raw/` | Role |
| --- | --- |
| `openrouter_frontend_models_day_20260812.json` | Daily model/variant demand |
| `openrouter_frontend_models_week_20260812.json` | Primary weekly demand, August 5–11 |
| `openrouter_models_catalog_20260812_trimmed.json` | Model identities, modalities, and token prices; unused API fields omitted |
| `openrouter_frontend_model_rankings_chart_20260812.json` | First-party weekly series for historical cross-validation |
| `openrouter_rankings_daily_20250101_20260604.json` | Historical daily top-50 observations and residual totals, January 1, 2025–June 4, 2026 |

The historical JSON contains data extracted from commit
[499c652ad7f91a31738bcb6969bbf7407b7c5eeb](https://github.com/Spectral-Finance/openrouter-rankings/commit/499c652ad7f91a31738bcb6969bbf7407b7c5eeb)
of `Spectral-Finance/openrouter-rankings`, a third-party archive rather than an
OpenRouter-controlled source. It preserves all 520 days of dates, model token
counts, residual counts, and daily totals; dashboard code and catalog metadata
are omitted. `parse_archive()` checks the JSON's pinned hash before loading it.
The provenance records the original archive's URL, hash, and retrieval time.
The upstream `meta.as_of` timestamp was not retained by that archive and is
unknown; its retrieval date is not a substitute for that timestamp.

The catalog retains all 535 listings in their original order, with only the
fields consumed by the analysis and supporting tables: identifiers (including
Hugging Face IDs), names, creation and expiration dates, context lengths,
input/output modalities, modality labels, tokenizers, and prompt/completion
prices. Retained values are unchanged. The manifest records both the original
response hash and the trimmed file hash, plus the field list.
`trim_catalog()` implements this selection; `collect` applies it to new catalog
captures before saving them. Descriptions, benchmarks, and unused API settings
are not included.

## Attribution and data license

Source: [OpenRouter rankings](https://openrouter.ai/rankings), captures retrieved
August 12, 2026. Rankings data is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on the
rankings page and in the [Data API documentation](https://openrouter.ai/docs/cookbook/administration/data-api).
Historical data was obtained via Spectral-Finance's archive cited above.
The JSON extraction and processed CSVs select, reshape, and aggregate the
source data as documented here and in `openrouter_demand.py`.

This attribution and license statement concerns rankings data, not the separate
Models API catalog. The catalog is attributed to OpenRouter in the retrieval
manifest and is not relicensed under this repository's MIT license.

## Processed files

All files below are in `processed/`.

| File | Contents |
| --- | --- |
| `selected_historical_daily_demand.csv` | Four historical dates used by the main analysis |
| `secondary_archive_cross_validation.csv` | Comparison with the first-party weekly series |
| `current_week_model_demand.csv` | Weekly model/variant counts and shares |
| `latest_day_model_demand.csv` | Daily model/variant counts and shares |
| `current_model_catalog.csv` | Catalog metadata |
| `official_weekly_top_models_series.csv` | First-party weekly chart in long form |
| `concentration_summary.csv` | Concentration diagnostics by measure and aggregation |
| `daily_top50_censoring_summary.csv` | Daily top-50 and residual-bucket diagnostics |
| `pseudo_freeze_retention.csv` | Observed retention of historical model cohorts |

The main analysis consumes `selected_historical_daily_demand.csv`.
The other files support reconstruction and source checks.

## Identifiers and fields

- `model_permaslug` identifies a model; `exact_model_variant` retains service
  variants such as free, batch, and thinking. The primary demand tables
  aggregate variants by permaslug.
- Historical listings are matched as recorded. The `other` bucket contains
  unnamed traffic and is excluded from common-model comparisons.
- Token counts are provider-native prompt plus completion tokens. Prices in
  the catalog are dollars per token.
- `heuristic_family` is a name-based diagnostic grouping, not a
  hardware-compatibility identifier.
- The weekly cross-validation excludes incomplete overlaps. Its residual
  differences can include daily top-50 censoring.

## Rebuild and verify

From the repository root:

```bash
python scripts/check_reproduction.py
python demand_data/openrouter/openrouter_demand.py verify
```

The rebuild runs in temporary space with explicit input paths. Direct
`openrouter_demand.py analyze` runs write to `processed/` and choose the newest
matching captures unless `--day-file`, `--week-file`, `--catalog-file`,
`--chart-file`, and `--archive-file` are supplied. Use the rebuild command
above to reproduce the committed data.

## Optional new captures

`python demand_data/openrouter/openrouter_demand.py collect` retrieves new
cross-sectional captures. Add `--daily` and set `OPENROUTER_API_KEY` to also
retrieve the authenticated daily history. The default history runs from
January 1, 2025 through yesterday in UTC; `--start-date` and `--end-date`
override it. Long ranges are split into contiguous requests of at most 366
dates to respect the Data API limit, with a separate file for each window.
New captures do not replace the inputs pinned for paper reproduction.
