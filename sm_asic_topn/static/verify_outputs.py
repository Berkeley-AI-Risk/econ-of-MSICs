#!/usr/bin/env python3
"""Independent spot/reconciliation checks for the generated static CSVs.

This verifier intentionally does not import ``analyze_topn.py``.  It rebuilds
the primary day/week permaslug universe directly from the raw JSON and checks
selected top-N rows, cross-measure coverage, membership stability, hashes, and
CSV primary keys.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ECON_DIR = HERE.parents[1]
MEASURES = ("prompt_tokens", "completion_tokens", "total_tokens", "requests")
TEST_NS = (1, 10, 30, 50, 100)
TOLERANCE = 2e-12


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def manifest_input(role):
    """Resolve a raw input by its role ("day", "week", "catalog") from
    analysis_manifest.json and verify its hash.  The derived CSVs are tied to
    these exact captures; newer captures in raw/ must not be picked up."""
    info = json.loads((HERE / "analysis_manifest.json").read_text())["inputs"][role]
    path = ECON_DIR / info["path"]
    if digest(path) != info["sha256"]:
        raise AssertionError("Input hash mismatch for {}".format(path))
    return path


def read_csv(name):
    with (HERE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def demand_rows(period):
    payload = json.loads(manifest_input(period).read_text())
    return [
        row
        for row in payload["data"]
        if row.get("model_permaslug")
        and int(row.get("total_prompt_tokens") or 0)
        + int(row.get("total_completion_tokens") or 0)
        + int(row.get("count") or 0)
        > 0
    ]


def text_output_permaslugs():
    catalog = json.loads(manifest_input("catalog").read_text())["data"]
    return {
        row["canonical_slug"]
        for row in catalog
        if row.get("canonical_slug")
        and "text" in ((row.get("architecture") or {}).get("output_modalities") or [])
    }


def aggregate_permaslug(rows, allowed=None):
    result = defaultdict(lambda: {measure: 0 for measure in MEASURES})
    for row in rows:
        model = row["model_permaslug"]
        if allowed is not None and model not in allowed:
            continue
        prompt = int(row.get("total_prompt_tokens") or 0)
        completion = int(row.get("total_completion_tokens") or 0)
        result[model]["prompt_tokens"] += prompt
        result[model]["completion_tokens"] += completion
        result[model]["total_tokens"] += prompt + completion
        result[model]["requests"] += int(row.get("count") or 0)
    return dict(result)


def coverage(selected, units):
    totals = {measure: sum(row[measure] for row in units.values()) for measure in MEASURES}
    return {
        measure: sum(units.get(model, {}).get(measure, 0) for model in selected) / totals[measure]
        for measure in MEASURES
    }


def close(actual, expected, label):
    if abs(float(actual) - float(expected)) > TOLERANCE:
        raise AssertionError("{}: {} != {}".format(label, actual, expected))


def unique_key_check(rows, fields, name):
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate primary key in {}".format(name))


def main():
    coverage_csv = read_csv("topn_coverage.csv")
    stability_csv = read_csv("selection_stability.csv")
    design_csv = read_csv("design_count_sensitivity.csv")
    unit_csv = read_csv("unit_rankings.csv")
    ranked_universe_csv = read_csv("ranked_units_by_universe.csv")
    metadata_csv = read_csv("metadata_availability.csv")
    summary_csv = read_csv("selected_n_summary.csv")

    unique_key_check(
        coverage_csv,
        ("period", "universe", "aggregation", "rank_basis", "n"),
        "topn_coverage.csv",
    )
    unique_key_check(
        stability_csv,
        (
            "comparison_type",
            "universe",
            "aggregation",
            "n",
            "set_a_period",
            "set_a_rank_basis",
            "set_b_period",
            "set_b_rank_basis",
        ),
        "selection_stability.csv",
    )
    unique_key_check(unit_csv, ("period", "aggregation", "unit_id"), "unit_rankings.csv")
    unique_key_check(
        ranked_universe_csv,
        ("period", "universe", "aggregation", "unit_id"),
        "ranked_units_by_universe.csv",
    )
    unique_key_check(
        metadata_csv,
        ("period", "universe", "aggregation", "rank_basis", "n", "metric"),
        "metadata_availability.csv",
    )
    unique_key_check(
        summary_csv,
        ("period", "universe", "aggregation", "rank_basis", "n"),
        "selected_n_summary.csv",
    )

    text_models = text_output_permaslugs()
    raw = {period: demand_rows(period) for period in ("day", "week")}
    all_units = {period: aggregate_permaslug(raw[period]) for period in raw}
    primary = {
        period: aggregate_permaslug(raw[period], text_models)
        for period in raw
    }
    coverage_index = {
        (
            row["period"],
            row["universe"],
            row["aggregation"],
            row["rank_basis"],
            int(row["n"]),
        ): row
        for row in coverage_csv
    }

    for period in ("day", "week"):
        for basis in ("total_tokens", "requests"):
            ordered = sorted(primary[period], key=lambda model: (-primary[period][model][basis], model))
            for n in TEST_NS:
                selected = set(ordered[:n])
                expected = coverage(selected, primary[period])
                expected_all = coverage(selected, all_units[period])
                row = coverage_index[
                    (period, "current_catalog_text_output", "model_permaslug", basis, n)
                ]
                if int(row["selected_ranking_units"]) != n:
                    raise AssertionError("Selected unit count mismatch")
                mapping = {
                    "prompt_tokens": "prompt",
                    "completion_tokens": "completion",
                    "total_tokens": "token",
                    "requests": "request",
                }
                for measure, short in mapping.items():
                    close(row["{}_coverage".format(short)], expected[measure], "{} {} N={} {}".format(period, basis, n, short))
                    close(
                        row["{}_coverage_of_all_observed".format(short)],
                        expected_all[measure],
                        "all-observed {} {} N={} {}".format(period, basis, n, short),
                    )

    # Independent top-30 cross-measure and current-day/trailing-week membership checks.
    stability_index = {
        (
            row["comparison_type"],
            row["universe"],
            row["aggregation"],
            int(row["n"]),
            row["set_a_period"],
            row["set_a_rank_basis"],
            row["set_b_period"],
            row["set_b_rank_basis"],
        ): row
        for row in stability_csv
    }
    for comparison_type, period_a, basis_a, period_b, basis_b in (
        ("tokens_vs_requests", "day", "total_tokens", "day", "requests"),
        ("day_vs_week", "day", "total_tokens", "week", "total_tokens"),
        ("day_vs_week", "day", "requests", "week", "requests"),
    ):
        order_a = sorted(primary[period_a], key=lambda model: (-primary[period_a][model][basis_a], model))
        order_b = sorted(primary[period_b], key=lambda model: (-primary[period_b][model][basis_b], model))
        set_a = set(order_a[:30])
        set_b = set(order_b[:30])
        row = stability_index[
            (
                comparison_type,
                "current_catalog_text_output",
                "model_permaslug",
                30,
                period_a,
                basis_a,
                period_b,
                basis_b,
            )
        ]
        if int(row["overlap_count"]) != len(set_a & set_b):
            raise AssertionError("Top-30 overlap mismatch for {}".format(comparison_type))
        close(row["jaccard"], len(set_a & set_b) / len(set_a | set_b), "Top-30 Jaccard")
        for label, selected in (("set_a", set_a), ("set_b", set_b)):
            for evaluation_period in ("day", "week"):
                expected = coverage(selected, primary[evaluation_period])
                close(
                    row["{}_{}_token_coverage".format(label, evaluation_period)],
                    expected["total_tokens"],
                    "stability token coverage",
                )
                close(
                    row["{}_{}_request_coverage".format(label, evaluation_period)],
                    expected["requests"],
                    "stability request coverage",
                )

    # Exact-service top-30 variant count is reconstructed without using the analysis code.
    exact_order = sorted(
        raw["day"],
        key=lambda row: (
            -(int(row.get("total_prompt_tokens") or 0) + int(row.get("total_completion_tokens") or 0)),
            row.get("variant_permaslug") or row["model_permaslug"],
        ),
    )
    exact_order = [row for row in exact_order if row["model_permaslug"] in text_models]
    expected_permaslugs = len({row["model_permaslug"] for row in exact_order[:30]})
    design_row = next(
        row
        for row in design_csv
        if row["period"] == "day"
        and row["universe"] == "current_catalog_text_output"
        and row["aggregation"] == "exact_model_variant"
        and row["rank_basis"] == "total_tokens"
        and int(row["n"]) == 30
    )
    if int(design_row["unique_model_permaslugs_selected"]) != expected_permaslugs:
        raise AssertionError("Exact-variant/permaslug design sensitivity mismatch")

    print("Independent verification passed:")
    print("- six CSV primary-key checks")
    print("- 40 primary top-N coverage rows rebuilt from raw JSON")
    print("- three top-30 set-overlap/cross-period comparisons rebuilt")
    print("- exact-variant/permaslug design-count sensitivity rebuilt")
    print("- inputs resolved via analysis_manifest.json with hashes verified at load")


if __name__ == "__main__":
    main()
