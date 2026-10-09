#!/usr/bin/env python3
"""Interactive companion to the Economics of MSICs analysis."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import streamlit as st


HERE = Path(__file__).resolve().parent
DEMAND_CSV = HERE / "sm_asic_topn" / "static" / "ranked_units_by_universe.csv"
EXPECTED_DEMAND_SHA256 = (
    "9c8409dbdff6837aa539a012b314e3421edd93d7af46a9998d91d31828822b97"
)
WEEK_END = "2026-08-11"
N_MODELS = 315


def logarithmic_slider_values(
    minimum: int,
    maximum: int,
    anchors: Sequence[int],
    steps_per_decade: int = 100,
) -> tuple[int, ...]:
    """Return dense log-spaced values plus exact economically useful anchors."""

    log_min = math.log10(minimum)
    log_max = math.log10(maximum)
    step_count = math.ceil((log_max - log_min) * steps_per_decade)
    values = {
        round(10 ** (log_min + index * (log_max - log_min) / step_count))
        for index in range(step_count + 1)
    }
    values.update(anchors)
    values.update((minimum, maximum))
    return tuple(sorted(value for value in values if minimum <= value <= maximum))


# About 100 stops per order of magnitude feels continuous while preserving
# exact paper values as selectable anchors.
NRE_OPTIONS = logarithmic_slider_values(
    1_000_000,
    2_000_000_000,
    anchors=(10_000_000, 50_000_000, 200_000_000, 500_000_000),
)
B_OPTIONS = logarithmic_slider_values(
    50_000_000_000,
    5_000_000_000_000,
    anchors=(100_000_000_000, 500_000_000_000, 1_000_000_000_000, 2_000_000_000_000),
)


@dataclass(frozen=True)
class Outcome:
    """The four quantities derived from one (S, NRE, B) scenario."""

    designs: int
    coverage: float
    savings: float
    unserved: float


@st.cache_data(show_spinner=False)
def load_demand_shares(path: Path = DEMAND_CSV) -> tuple[float, ...]:
    """Load and validate the pinned weekly 315-model demand distribution."""

    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual_hash == EXPECTED_DEMAND_SHA256, (
        "Pinned demand data changed: expected SHA-256 "
        f"{EXPECTED_DEMAND_SHA256}, got {actual_hash}"
    )

    rows = []
    seen_units = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["period"] != "week"
                or row["universe"] != "current_catalog_text_output"
                or row["aggregation"] != "model_permaslug"
            ):
                continue

            assert row["period_end_date"] == WEEK_END, (
                f"Expected week ending {WEEK_END}, got {row['period_end_date']}"
            )
            unit_id = row["unit_id"]
            assert unit_id not in seen_units, f"Duplicate demand row for {unit_id}"
            seen_units.add(unit_id)

            share = float(row["token_share_within_universe"])
            assert math.isfinite(share) and share >= 0, (
                f"Invalid token share for {unit_id}: {share}"
            )
            rows.append(share)

    assert len(rows) == N_MODELS, f"Expected {N_MODELS} rows, got {len(rows)}"
    assert math.isclose(math.fsum(rows), 1.0, rel_tol=0.0, abs_tol=1e-9), (
        f"Demand shares must sum to 1; got {math.fsum(rows):.12f}"
    )
    return tuple(sorted(rows, reverse=True))


def evaluate(D: Sequence[float], S: float, NRE: int, B: int) -> Outcome:
    """Apply S * D_M * B > NRE; D contains fractional shares, not percentages.

    S is also a fraction; NRE and B are dollars. Savings is net of one NRE
    payment per selected model and includes flexible serving of the remainder.
    """

    selected = [D_M for D_M in D if S * D_M * B > NRE]
    coverage = math.fsum(selected)
    designs = len(selected)
    savings = S * coverage - designs * NRE / B
    return Outcome(
        designs=designs,
        coverage=coverage,
        savings=savings,
        unserved=1.0 - coverage,
    )


def assert_figure_fidelity(D: Sequence[float]) -> None:
    """Check every adoption-figure scenario and the lower-baseline stress case."""

    figure_cases = (
        # B, NRE, N, coverage %, coverage decimals, savings %, savings decimals
        (500_000_000_000, 10_000_000, 196, 99.93, 2, 84.5, 1),
        (500_000_000_000, 50_000_000, 134, 99.59, 2, 83.3, 1),
        (500_000_000_000, 200_000_000, 86, 98.34, 2, 80.1, 1),
        (500_000_000_000, 500_000_000, 60, 96.25, 2, 75.8, 1),
        (1_000_000_000_000, 10_000_000, 221, 99.97, 2, 84.8, 1),
        (1_000_000_000_000, 50_000_000, 157, 99.78, 2, 84.0, 1),
        (1_000_000_000_000, 200_000_000, 111, 99.17, 2, 82.1, 1),
        (1_000_000_000_000, 500_000_000, 81, 98.09, 2, 79.3, 1),
        (2_000_000_000_000, 10_000_000, 241, 99.99, 2, 84.9, 1),
        (2_000_000_000_000, 50_000_000, 185, 99.90, 2, 84.4, 1),
        (2_000_000_000_000, 200_000_000, 134, 99.59, 2, 83.3, 1),
        (2_000_000_000_000, 500_000_000, 102, 98.94, 2, 81.5, 1),
    )
    for (
        B,
        NRE,
        expected_n,
        expected_cov,
        cov_dp,
        expected_save,
        save_dp,
    ) in figure_cases:
        result = evaluate(D, 0.85, NRE, B)
        assert result.designs == expected_n, (
            f"Fidelity check failed for B={B}, NRE={NRE}: "
            f"expected {expected_n} designs, got {result.designs}"
        )
        assert round(100 * result.coverage, cov_dp) == expected_cov, (
            f"Fidelity check failed for B={B}, NRE={NRE}: "
            f"expected {expected_cov}% coverage, got {100 * result.coverage:.8f}%"
        )
        assert round(100 * result.savings, save_dp) == expected_save, (
            f"Fidelity check failed for B={B}, NRE={NRE}: "
            f"expected {expected_save}% savings, got {100 * result.savings:.8f}%"
        )

    stress = evaluate(D, 0.85, 500_000_000, 100_000_000_000)
    assert stress.designs == 26, (
        f"Stress-case fidelity check expected 26 designs, got {stress.designs}"
    )
    assert round(100 * stress.coverage, 1) == 86.5, (
        "Stress-case fidelity check expected 86.5% coverage, got "
        f"{100 * stress.coverage:.8f}%"
    )


def format_dollars(value: int) -> str:
    """Format exact control values compactly."""

    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.3g}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.3g}B"
    return f"${value / 1_000_000:.3g}M"


def render_figure_bar(outcome: Outcome) -> None:
    """Render the current scenario as a stacked bar of demand coverage."""

    coverage_percent = 100 * outcome.coverage
    flexible_percent = 100 * outcome.unserved
    coverage_label = (
        f"{coverage_percent:.2f}"
        if flexible_percent < 0.1
        else f"{coverage_percent:.1f}"
    )
    if 0 < flexible_percent < 0.005:
        flexible_label = "<0.01% on flexible"
    elif flexible_percent < 0.1:
        flexible_label = f"{flexible_percent:.2f}% on flexible"
    else:
        flexible_label = f"{flexible_percent:.1f}% on flexible"
    savings_label = f"saves {100 * max(0.0, outcome.savings):.0f}% vs all-flexible"

    bar_data = [
        {
            "scenario": "Current setting",
            "hardware": "tokens served on dedicated MSICs",
            "share": outcome.coverage,
            "order": 1,
        },
        {
            "scenario": "Current setting",
            "hardware": "tokens served on flexible accelerators",
            "share": outcome.unserved,
            "order": 2,
        },
    ]
    labels = [
        {
            "scenario": "Current setting",
            "inside_x": 0.015,
            "middle_x": 0.5,
            "right_x": 0.99,
            "coverage_label": (
                f"{outcome.designs} designs · {coverage_label}% of tokens"
            ),
            "flexible_label": flexible_label,
            "savings_label": savings_label,
        }
    ]
    chart_spec = {
        "height": 180,
        "padding": {"top": 12, "bottom": 8, "left": 4, "right": 4},
        "layer": [
            {
                "data": {"values": bar_data},
                "mark": {
                    "type": "bar",
                    "height": 62,
                    "stroke": "white",
                    "strokeWidth": 2,
                },
                "encoding": {
                    "x": {
                        "field": "share",
                        "type": "quantitative",
                        "stack": "zero",
                        "title": (
                            "Share of observed tokens "
                            "(315 text-output models on OpenRouter)"
                        ),
                        "scale": {"domain": [0, 1]},
                        "axis": {
                            "format": ".0%",
                            "values": [0, 0.25, 0.5, 0.75, 1],
                            "offset": 6,
                            "labelPadding": 8,
                            "titlePadding": 14,
                        },
                    },
                    "y": {
                        "field": "scenario",
                        "type": "nominal",
                        "axis": None,
                    },
                    "color": {
                        "field": "hardware",
                        "type": "nominal",
                        "scale": {
                            "domain": [
                                "tokens served on dedicated MSICs",
                                "tokens served on flexible accelerators",
                            ],
                            "range": ["#2a78d6", "#eb6834"],
                        },
                        "legend": {
                            "title": None,
                            "orient": "top",
                            "direction": "horizontal",
                            "columns": 2,
                            "labelFontSize": 11,
                            "labelLimit": 300,
                        },
                    },
                    "order": {
                        "field": "order",
                        "type": "quantitative",
                        "sort": "ascending",
                    },
                    "tooltip": [
                        {
                            "field": "hardware",
                            "type": "nominal",
                            "title": "Serving hardware",
                        },
                        {
                            "field": "share",
                            "type": "quantitative",
                            "title": "Share of tokens",
                            "format": ".4%",
                        },
                    ],
                },
            },
            {
                "data": {"values": labels},
                "mark": {
                    "type": "text",
                    "align": "left",
                    "baseline": "middle",
                    "color": "white",
                    "fontSize": 13,
                    "fontWeight": "bold",
                },
                "encoding": {
                    "x": {"field": "inside_x", "type": "quantitative"},
                    "y": {"field": "scenario", "type": "nominal"},
                    "text": {"field": "coverage_label"},
                },
            },
            {
                "data": {"values": labels},
                "mark": {
                    "type": "text",
                    "align": "right",
                    "baseline": "bottom",
                    "dy": -39,
                    "color": "#eb6834",
                    "fontSize": 12,
                },
                "encoding": {
                    "x": {"field": "right_x", "type": "quantitative"},
                    "y": {"field": "scenario", "type": "nominal"},
                    "text": {"field": "flexible_label"},
                },
            },
            {
                "data": {"values": labels},
                "mark": {
                    "type": "text",
                    "align": "center",
                    "baseline": "bottom",
                    "dy": -39,
                    "color": "#898781",
                    "fontSize": 12,
                },
                "encoding": {
                    "x": {"field": "middle_x", "type": "quantitative"},
                    "y": {"field": "scenario", "type": "nominal"},
                    "text": {"field": "savings_label"},
                },
            },
        ],
    }
    st.vega_lite_chart(spec=chart_spec, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Economics of MSICs",
        page_icon="📈",
        layout="wide",
    )

    D = load_demand_shares()
    assert_figure_fidelity(D)

    st.title("Economics of MSICs")
    st.write(
        "Explore the three inputs in the \"Economics of MSICs\" "
        "section: in a world with a frozen whitelist of AI models, a model "
        "receives a dedicated MSIC exactly when its expected lifetime cost "
        "savings exceed the design cost:"
    )
    st.latex(r"S \times D_M \times B > \mathrm{NRE}")

    with st.sidebar:
        st.header("Parameters")
        S_percent = st.slider(
            "MSIC savings rate S",
            min_value=50.0,
            max_value=99.0,
            value=85.0,
            step=0.1,
            format="%.1f%%",
        )
        st.caption("The paper considers a savings range of 60–95%; the adoption figure uses 85%.")
        NRE = st.select_slider(
            "MSIC development cost NRE",
            options=NRE_OPTIONS,
            value=500_000_000,
            format_func=format_dollars,
        )
        B = st.select_slider(
            "Lifetime all-flexible serving cost B",
            options=B_OPTIONS,
            value=500_000_000_000,
            format_func=format_dollars,
        )

    S = S_percent / 100
    outcome = evaluate(D, S, NRE, B)
    displayed_savings = max(0.0, outcome.savings)

    metric_columns = st.columns(3)
    metric_columns[0].metric("MSIC designs", f"{outcome.designs}")
    metric_columns[1].metric("Tokens on MSICs", f"{100 * outcome.coverage:.2f}%")
    metric_columns[2].metric(
        "Savings vs all-flexible", f"{100 * displayed_savings:.1f}%"
    )
    st.markdown(
        f"**≤ {100 * outcome.unserved:.2f}% of tokens unserved in a pure-MSIC world**"
    )
    st.caption(
        "This is an upper bound, not a prediction: switching and post-switch MSIC "
        "creation can only reduce the unserved share."
    )

    st.subheader("Where inference runs during the pause")
    render_figure_bar(outcome)
    NRE_label = format_dollars(NRE).replace("$", r"\$")
    B_label = format_dollars(B).replace("$", r"\$")
    st.caption(
        f"Current setting: S = {S:.1%}, NRE = {NRE_label}, "
        f"B = {B_label}."
    )

    st.divider()
    st.markdown(
        '> This is a deliberately simple back-of-the-envelope model: thresholds '
        'applied to one week of OpenRouter token shares (August 5–11, 2026) as a '
        'proxy for expected lifetime demand shares. Token shares are not a welfare '
        'measure, S, NRE, and B are scenario inputs rather than measured quantities, '
        'and nothing here models demand shifts over time. See the "Economics of '
        'MSICs" section for the full analysis and caveats.'
    )
    st.caption(
        "Pinned data: 315 text-output models with observed OpenRouter traffic, "
        "week ending August 11, 2026. No model names are displayed."
    )


if __name__ == "__main__":
    main()
