# Economics of MSICs

Code and data accompanying the **Economics of MSICs** section of
*The Feasibility of a Hardwired Pause of Frontier AI Training*.

This repository reproduces the MSIC adoption figure and the calculations based
on archived OpenRouter data. The paper also contains hand-worked illustrations
of utilization and de-whitelisting, which are not simulated here. Demand shares
are a token-based proxy; the results are illustrative scenarios, not forecasts
or estimates of consumer welfare.

Code and documentation written by Claude Code and Codex at the instruction of the authors.

## Reproduce the analysis

Use **Python 3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.
Installation needs internet access; reproduction runs offline without API keys.

Outputs are written to `build/`:

- `econ_fig_1.pdf`, `econ_fig_1.png`, and `econ_fig_1.jpg`: adoption figure.
- `figure-grid.csv` and `figure-grid.json`: numerical values for each figure cell.
- `report.txt`: computed results and robustness checks.

## Run the interactive app

Explore the [hosted app](https://econ-of-msics.streamlit.app), or run locally:

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. The controls vary `S`, `NRE`, and `B`.

## Inspect the code

- [msic_analysis.py](msic_analysis.py): input validation, calculations, figure,
  and report. The figure parameters are `S`, `NRE_VALUES`, and `B_VALUES`.
- [streamlit_app.py](streamlit_app.py): app controls and scenario evaluation.
- [Code and data conventions](docs/model-and-data.md): function entry points,
  CSV selection, and pricing treatment.
- [OpenRouter sources](demand_data/openrouter/README.md): raw captures and
  historical-data provenance.
- [Demand tables](sm_asic_topn/static/README.md): CSV fields, aggregation, and
  data-generation commands.

| Paper notation | Python representation |
| --- | --- |
| M | `M`: model identifier |
| D_M | `D[M]` or scalar `D_M` |
| S | `S` |
| NRE | `NRE` |
| B | `B` |

Python inputs use fractions for `D_M` and `S` (85% is `0.85`) and dollars
for `NRE` and `B`. Output fields ending in `_percent` use percentages.

## Run the checks

```bash
python -m unittest discover -v
python sm_asic_topn/static/verify_outputs.py
python demand_data/openrouter/openrouter_demand.py verify
python scripts/check_reproduction.py
```

The last command rebuilds derived data in a temporary directory without
overwriting the committed files. See [Validation](docs/validation.md) for
test coverage and comparison tolerances.

Numerical tables and input hashes are checked for reproducibility. Figure
metadata, font rendering, and image bytes can differ across environments.

## License

The code and original documentation are available under the [MIT License](LICENSE).
Archived third-party source material is not relicensed by this grant; see
[source provenance](demand_data/openrouter/README.md).
