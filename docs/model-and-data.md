# Code and data conventions

## Calculation entry points

In [msic_analysis.py](../msic_analysis.py):

- `verify_inputs()` validates the three pinned input hashes.
- `load_demand()` returns model-keyed demand shares and prompt/completion counts.
- `msic_set(D, NRE, B)` returns selected model identifiers using the module's `S`.
- `panel_rows(D, B)` returns design counts, coverage, and savings for `NRE_VALUES`;
  its coverage and savings values are percentages.
- `price_weighted_check(D, tokens)` evaluates both pricing conventions.
- `demand_shift_analysis()` returns date pairs, common-model counts, and shifts.
- `make_figure(D, stem)` writes the figure; `main()` runs the full report.

In [streamlit_app.py](../streamlit_app.py), `evaluate(D, S, NRE, B)` accepts a
sequence of fractional demand shares and returns an `Outcome`. Its `coverage`,
`savings`, and `unserved` fields are fractions; `designs` is a count.

## Input selection

The main analysis reads `ranked_units_by_universe.csv` with
`period=week`, `universe=current_catalog_text_output`, and
`aggregation=model_permaslug`. The loader checks the week ending
`2026-08-11`, unique model identifiers, and finite nonnegative values.

The pricing catalog and historical CSV are also pinned in
`EXPECTED_SHA256`. Call `verify_inputs()` before loading data when using the
module directly; the reproduction command calls it automatically.

## Pricing and historical-data handling

`_catalog_prices()` constructs exact-listing and base-model price maps.
Negative, nonfinite, or unparsable price pairs are excluded. The
standard-price calculation omits models with missing or zero base prices.
The variant-specific calculation uses exact-listing prices where available;
otherwise free variants receive zero and other variants fall back to the
base price or zero.

Historical comparisons use raw listing identifiers, excluding the `other`
bucket, and normalize over identifiers present at both dates with positive
later demand. This differs from the primary demand table's service-variant
aggregation.

See the [source inventory](../demand_data/openrouter/README.md) and
[CSV documentation](../sm_asic_topn/static/README.md) for file-level details.
