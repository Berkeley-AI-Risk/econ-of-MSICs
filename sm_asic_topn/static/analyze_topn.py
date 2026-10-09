#!/usr/bin/env python3
"""Static top-N OpenRouter demand accounting for the SM-ASIC analysis.

This script deliberately computes demand *coverage*, not economic inefficiency.
It uses the pinned current-day, trailing-week, and catalog JSON snapshots already
archived in ``demand_data/openrouter/raw``.  Python's standard
library is sufficient.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ECON_DIR = HERE.parents[1]
RAW_DIR = ECON_DIR / "demand_data" / "openrouter" / "raw"

MEASURES = ("prompt_tokens", "completion_tokens", "total_tokens", "requests")
RANK_BASES = ("total_tokens", "requests")
PERIODS = ("day", "week")
AGGREGATIONS = ("exact_model_variant", "model_permaslug")
UNIVERSES = ("current_catalog_text_output", "all_observed")
NON_TEXT_ACTIVITY_FIELDS = (
    "num_media_prompt",
    "num_media_completion",
    "image_output_requests",
    "num_video_prompt",
    "video_output_seconds",
    "rerank_documents",
    "stt_transcript_characters",
    "num_audio_prompt",
)
POLICY_GRID = (1, 2, 5, 10, 20, 25, 30, 50, 75, 100, 200)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_int(value):
    if value in (None, ""):
        return 0
    return int(value)


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def catalog_variant(catalog_id):
    if ":" not in catalog_id:
        return "standard"
    suffix = catalog_id.rsplit(":", 1)[1]
    return suffix if suffix in {"free", "batch", "thinking"} else "standard"


def joined(values):
    return "|".join(sorted({str(value) for value in values if value not in (None, "")}))


def load_inputs():
    # Reproduction must not silently switch to a newer capture in raw/.
    manifest = json.loads((HERE / "analysis_manifest.json").read_text(encoding="utf-8"))
    paths = {}
    for role in ("day", "week", "catalog"):
        source = manifest["inputs"][role]
        path = ECON_DIR / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError("Pinned input hash mismatch: {}".format(path))
        paths[role] = path
    demand = {}
    for period in PERIODS:
        payload = json.loads(paths[period].read_text(encoding="utf-8"))
        rows = payload["data"]
        # The source currently contains one blank, zero-demand sentinel.  It is
        # not a candidate model and is excluded from rank/design counts.
        demand[period] = [
            row
            for row in rows
            if row.get("model_permaslug")
            and (
                as_int(row.get("total_prompt_tokens"))
                + as_int(row.get("total_completion_tokens"))
                + as_int(row.get("count"))
                > 0
            )
        ]
    catalog_payload = json.loads(paths["catalog"].read_text(encoding="utf-8"))
    catalog = catalog_payload["data"]
    return paths, demand, catalog


def index_catalog(catalog):
    by_canonical = defaultdict(list)
    for row in catalog:
        canonical = row.get("canonical_slug")
        if canonical:
            by_canonical[canonical].append(row)
    return by_canonical


def metadata(permaslug, observed_variants, catalog_rows, aggregation):
    variants = set(observed_variants)
    exact_matches = [row for row in catalog_rows if catalog_variant(row.get("id", "")) in variants]
    architecture_rows = [row for row in catalog_rows if isinstance(row.get("architecture"), dict)]
    input_modalities = {
        modality
        for row in architecture_rows
        for modality in (row["architecture"].get("input_modalities") or [])
    }
    output_modalities = {
        modality
        for row in architecture_rows
        for modality in (row["architecture"].get("output_modalities") or [])
    }
    modalities = {row["architecture"].get("modality") for row in architecture_rows}
    tokenizers = {row["architecture"].get("tokenizer") for row in architecture_rows}
    contexts = [as_int(row.get("context_length")) for row in catalog_rows if row.get("context_length") is not None]
    hf_ids = {row.get("hugging_face_id") for row in catalog_rows if row.get("hugging_face_id")}
    return {
        "catalog_match": int(bool(catalog_rows)),
        "catalog_row_count": len(catalog_rows),
        "catalog_exact_variant_match": int(bool(exact_matches)) if aggregation == "exact_model_variant" else "",
        "catalog_exact_variant_match_count": len(exact_matches) if aggregation == "exact_model_variant" else "",
        "hugging_face_id_available": int(bool(hf_ids)),
        "hugging_face_ids": joined(hf_ids),
        "architecture_object_available": int(bool(architecture_rows)),
        "catalog_modality": joined(modalities),
        "catalog_input_modalities": joined(input_modalities),
        "catalog_output_modalities": joined(output_modalities),
        "catalog_tokenizer": joined(tokenizers),
        "tokenizer_known_non_other": int(any(tokenizer not in (None, "", "Other") for tokenizer in tokenizers)),
        "modality_metadata_available": int(bool(input_modalities or output_modalities)),
        "text_only_input_capable": int(input_modalities == {"text"}),
        "non_text_input_capable": int(any(modality != "text" for modality in input_modalities)),
        "text_output_capable": int("text" in output_modalities),
        "non_text_output_capable": int(any(modality != "text" for modality in output_modalities)),
        # The OpenRouter catalog has context/tokenizer/modality fields, but no
        # structured weights, total/active parameters, layers, hidden width,
        # expert layout, precision, or memory footprint.
        "structured_model_size_available": 0,
        "context_length_min": min(contexts) if contexts else "",
        "context_length_max": max(contexts) if contexts else "",
        "_tokenizer_set": {value for value in tokenizers if value},
        "_modality_set": {value for value in modalities if value},
    }


def aggregate_units(raw_rows, aggregation, catalog_index):
    groups = defaultdict(list)
    for raw in raw_rows:
        permaslug = raw["model_permaslug"]
        variant = raw.get("variant") or "standard"
        exact_id = raw.get("variant_permaslug") or (
            permaslug if variant == "standard" else "{}:{}".format(permaslug, variant)
        )
        unit_id = exact_id if aggregation == "exact_model_variant" else permaslug
        groups[unit_id].append(raw)

    units = []
    for unit_id, members in groups.items():
        permaslugs = {row["model_permaslug"] for row in members}
        if len(permaslugs) != 1:
            raise AssertionError("Aggregation unit spans permaslugs: {}".format(unit_id))
        permaslug = next(iter(permaslugs))
        variants = {row.get("variant") or "standard" for row in members}
        exact_ids = {
            row.get("variant_permaslug")
            or (permaslug if (row.get("variant") or "standard") == "standard" else "{}:{}".format(permaslug, row.get("variant")))
            for row in members
        }
        prompt = sum(as_int(row.get("total_prompt_tokens")) for row in members)
        completion = sum(as_int(row.get("total_completion_tokens")) for row in members)
        requests = sum(as_int(row.get("count")) for row in members)
        non_text_activity = sum(
            as_int(row.get(field)) for row in members for field in NON_TEXT_ACTIVITY_FIELDS
        )
        row = {
            "aggregation": aggregation,
            "unit_id": unit_id,
            "model_permaslug": permaslug,
            "variant": next(iter(variants)) if aggregation == "exact_model_variant" else "aggregated",
            "observed_variants": joined(variants),
            "observed_exact_variant_count": len(exact_ids),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "requests": requests,
            "tokens_per_request": safe_div(prompt + completion, requests),
            "completion_fraction_of_tokens": safe_div(completion, prompt + completion),
            "free_service_variant": int("free" in variants),
            "batch_service_variant": int("batch" in variants),
            "thinking_service_variant": int("thinking" in variants),
            "observed_non_text_activity": int(non_text_activity > 0),
            "observed_non_text_activity_units": non_text_activity,
            "_exact_ids": exact_ids,
            "_permaslugs": {permaslug},
        }
        row.update(metadata(permaslug, variants, catalog_index.get(permaslug, []), aggregation))
        units.append(row)

    totals = {measure: sum(row[measure] for row in units) for measure in MEASURES}
    for measure in MEASURES:
        share_name = {
            "prompt_tokens": "prompt_share",
            "completion_tokens": "completion_share",
            "total_tokens": "token_share",
            "requests": "request_share",
        }[measure]
        for row in units:
            row[share_name] = safe_div(row[measure], totals[measure])

    for basis, rank_name in (("total_tokens", "rank_tokens"), ("requests", "rank_requests")):
        ordered = sorted(units, key=lambda row: (-row[basis], row["unit_id"]))
        for rank, row in enumerate(ordered, 1):
            row[rank_name] = rank
    return units


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def public_unit_rows(datasets):
    fields = [
        "period",
        "period_end_date",
        "aggregation",
        "unit_id",
        "model_permaslug",
        "variant",
        "observed_variants",
        "observed_exact_variant_count",
        "rank_tokens",
        "rank_requests",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "requests",
        "prompt_share",
        "completion_share",
        "token_share",
        "request_share",
        "tokens_per_request",
        "completion_fraction_of_tokens",
        "free_service_variant",
        "batch_service_variant",
        "thinking_service_variant",
        "observed_non_text_activity",
        "observed_non_text_activity_units",
        "catalog_match",
        "catalog_row_count",
        "catalog_exact_variant_match",
        "catalog_exact_variant_match_count",
        "hugging_face_id_available",
        "hugging_face_ids",
        "architecture_object_available",
        "catalog_modality",
        "catalog_input_modalities",
        "catalog_output_modalities",
        "catalog_tokenizer",
        "tokenizer_known_non_other",
        "modality_metadata_available",
        "text_only_input_capable",
        "non_text_input_capable",
        "text_output_capable",
        "non_text_output_capable",
        "structured_model_size_available",
        "in_current_catalog_text_output_universe",
        "context_length_min",
        "context_length_max",
        "source_file",
    ]
    rows = []
    for (period, aggregation), data in datasets.items():
        for row in data["units"]:
            output = dict(row)
            output["in_current_catalog_text_output_universe"] = int(
                bool(row["catalog_match"] and row["text_output_capable"])
            )
            output.update(
                {
                    "period": period,
                    "period_end_date": data["period_end_date"],
                    "source_file": data["source_file"],
                }
            )
            rows.append(output)
    rows.sort(key=lambda row: (row["period"], row["aggregation"], row["rank_tokens"], row["unit_id"]))
    return rows, fields


def universe_unit_rows(datasets):
    fields = [
        "period",
        "period_end_date",
        "universe",
        "aggregation",
        "unit_id",
        "model_permaslug",
        "variant",
        "observed_variants",
        "observed_exact_variant_count",
        "rank_tokens_within_universe",
        "rank_requests_within_universe",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "requests",
        "prompt_share_within_universe",
        "completion_share_within_universe",
        "token_share_within_universe",
        "request_share_within_universe",
        "prompt_share_of_all_observed",
        "completion_share_of_all_observed",
        "token_share_of_all_observed",
        "request_share_of_all_observed",
        "tokens_per_request",
        "completion_fraction_of_tokens",
        "free_service_variant",
        "batch_service_variant",
        "thinking_service_variant",
        "catalog_match",
        "catalog_exact_variant_match",
        "hugging_face_id_available",
        "catalog_modality",
        "catalog_input_modalities",
        "catalog_output_modalities",
        "catalog_tokenizer",
        "tokenizer_known_non_other",
        "non_text_input_capable",
        "text_output_capable",
        "non_text_output_capable",
        "structured_model_size_available",
        "source_file",
    ]
    output_rows = []
    for (period, aggregation), data in datasets.items():
        all_units = data["units"]
        all_totals = {measure: sum(row[measure] for row in all_units) for measure in MEASURES}
        for universe in UNIVERSES:
            units = units_for_universe(all_units, universe)
            totals = {measure: sum(row[measure] for row in units) for measure in MEASURES}
            token_rank = {
                row["unit_id"]: rank
                for rank, row in enumerate(sorted(units, key=lambda item: (-item["total_tokens"], item["unit_id"])), 1)
            }
            request_rank = {
                row["unit_id"]: rank
                for rank, row in enumerate(sorted(units, key=lambda item: (-item["requests"], item["unit_id"])), 1)
            }
            for row in units:
                out = dict(row)
                out.update(
                    {
                        "period": period,
                        "period_end_date": data["period_end_date"],
                        "universe": universe,
                        "rank_tokens_within_universe": token_rank[row["unit_id"]],
                        "rank_requests_within_universe": request_rank[row["unit_id"]],
                        "prompt_share_within_universe": safe_div(row["prompt_tokens"], totals["prompt_tokens"]),
                        "completion_share_within_universe": safe_div(row["completion_tokens"], totals["completion_tokens"]),
                        "token_share_within_universe": safe_div(row["total_tokens"], totals["total_tokens"]),
                        "request_share_within_universe": safe_div(row["requests"], totals["requests"]),
                        "prompt_share_of_all_observed": safe_div(row["prompt_tokens"], all_totals["prompt_tokens"]),
                        "completion_share_of_all_observed": safe_div(row["completion_tokens"], all_totals["completion_tokens"]),
                        "token_share_of_all_observed": safe_div(row["total_tokens"], all_totals["total_tokens"]),
                        "request_share_of_all_observed": safe_div(row["requests"], all_totals["requests"]),
                        "source_file": data["source_file"],
                    }
                )
                output_rows.append(out)
    output_rows.sort(
        key=lambda row: (
            row["period"],
            row["universe"],
            row["aggregation"],
            row["rank_tokens_within_universe"],
            row["unit_id"],
        )
    )
    return output_rows, fields


def units_for_universe(units, universe):
    if universe == "all_observed":
        return list(units)
    if universe == "current_catalog_text_output":
        return [row for row in units if row["catalog_match"] and row["text_output_capable"]]
    raise ValueError("Unknown universe {}".format(universe))


def selected_property_counts(selected):
    properties = (
        "catalog_match",
        "hugging_face_id_available",
        "tokenizer_known_non_other",
        "modality_metadata_available",
        "structured_model_size_available",
        "text_only_input_capable",
        "non_text_input_capable",
        "text_output_capable",
        "non_text_output_capable",
        "free_service_variant",
        "batch_service_variant",
        "thinking_service_variant",
        "observed_non_text_activity",
    )
    return {prop: sum(int(row.get(prop) or 0) for row in selected) for prop in properties}


def topn_outputs(datasets):
    coverage_rows = []
    design_rows = []
    metadata_rows = []
    selected_summary_rows = []

    for (period, aggregation), data in datasets.items():
        all_units = data["units"]
        all_totals = {measure: sum(row[measure] for row in all_units) for measure in MEASURES}
        for universe in UNIVERSES:
            units = units_for_universe(all_units, universe)
            totals = {measure: sum(row[measure] for row in units) for measure in MEASURES}
            for basis in RANK_BASES:
                ordered = sorted(units, key=lambda row: (-row[basis], row["unit_id"]))
                cumulative = {measure: 0 for measure in MEASURES}
                selected = []
                policy_ns = set(POLICY_GRID) | {len(ordered)}
                for n, row in enumerate(ordered, 1):
                    selected.append(row)
                    for measure in MEASURES:
                        cumulative[measure] += row[measure]
                    permaslugs = set().union(*(item["_permaslugs"] for item in selected))
                    exact_ids = set().union(*(item["_exact_ids"] for item in selected))
                    properties = selected_property_counts(selected)
                    output = {
                    "period": period,
                    "period_end_date": data["period_end_date"],
                    "universe": universe,
                    "aggregation": aggregation,
                    "rank_basis": basis,
                    "n": n,
                    "available_ranking_units": len(ordered),
                    "selected_ranking_units": n,
                    "unique_model_permaslugs_selected": len(permaslugs),
                    "observed_exact_service_variants_selected": len(exact_ids),
                    "prompt_tokens_selected": cumulative["prompt_tokens"],
                    "completion_tokens_selected": cumulative["completion_tokens"],
                    "total_tokens_selected": cumulative["total_tokens"],
                    "requests_selected": cumulative["requests"],
                    "universe_prompt_tokens": totals["prompt_tokens"],
                    "universe_completion_tokens": totals["completion_tokens"],
                    "universe_total_tokens": totals["total_tokens"],
                    "universe_requests": totals["requests"],
                    "universe_prompt_share_of_all_observed": safe_div(totals["prompt_tokens"], all_totals["prompt_tokens"]),
                    "universe_completion_share_of_all_observed": safe_div(totals["completion_tokens"], all_totals["completion_tokens"]),
                    "universe_token_share_of_all_observed": safe_div(totals["total_tokens"], all_totals["total_tokens"]),
                    "universe_request_share_of_all_observed": safe_div(totals["requests"], all_totals["requests"]),
                    "prompt_coverage": safe_div(cumulative["prompt_tokens"], totals["prompt_tokens"]),
                    "completion_coverage": safe_div(cumulative["completion_tokens"], totals["completion_tokens"]),
                    "token_coverage": safe_div(cumulative["total_tokens"], totals["total_tokens"]),
                    "request_coverage": safe_div(cumulative["requests"], totals["requests"]),
                    "prompt_omitted_share": 1.0 - safe_div(cumulative["prompt_tokens"], totals["prompt_tokens"]),
                    "completion_omitted_share": 1.0 - safe_div(cumulative["completion_tokens"], totals["completion_tokens"]),
                    "token_omitted_share": 1.0 - safe_div(cumulative["total_tokens"], totals["total_tokens"]),
                    "request_omitted_share": 1.0 - safe_div(cumulative["requests"], totals["requests"]),
                    "prompt_coverage_of_all_observed": safe_div(cumulative["prompt_tokens"], all_totals["prompt_tokens"]),
                    "completion_coverage_of_all_observed": safe_div(cumulative["completion_tokens"], all_totals["completion_tokens"]),
                    "token_coverage_of_all_observed": safe_div(cumulative["total_tokens"], all_totals["total_tokens"]),
                    "request_coverage_of_all_observed": safe_div(cumulative["requests"], all_totals["requests"]),
                    "prompt_omitted_share_of_all_observed": 1.0 - safe_div(cumulative["prompt_tokens"], all_totals["prompt_tokens"]),
                    "completion_omitted_share_of_all_observed": 1.0 - safe_div(cumulative["completion_tokens"], all_totals["completion_tokens"]),
                    "token_omitted_share_of_all_observed": 1.0 - safe_div(cumulative["total_tokens"], all_totals["total_tokens"]),
                    "request_omitted_share_of_all_observed": 1.0 - safe_div(cumulative["requests"], all_totals["requests"]),
                    "completion_fraction_within_selected_tokens": safe_div(cumulative["completion_tokens"], cumulative["total_tokens"]),
                    "free_units_selected": properties["free_service_variant"],
                    "batch_units_selected": properties["batch_service_variant"],
                    "thinking_units_selected": properties["thinking_service_variant"],
                    "non_text_input_capable_units_selected": properties["non_text_input_capable"],
                    "non_text_output_capable_units_selected": properties["non_text_output_capable"],
                    "catalog_unmatched_units_selected": n - properties["catalog_match"],
                    "hf_id_available_units_selected": properties["hugging_face_id_available"],
                    "hardware_compatibility_mapping_available": 0,
                    }
                    coverage_rows.append(output)

                    tokenizer_labels = set().union(*(item["_tokenizer_set"] for item in selected))
                    modality_labels = set().union(*(item["_modality_set"] for item in selected))
                    design_rows.append(
                    {
                        "period": period,
                        "period_end_date": data["period_end_date"],
                        "universe": universe,
                        "aggregation": aggregation,
                        "rank_basis": basis,
                        "n": n,
                        "selected_ranking_units": n,
                        "unique_model_permaslugs_selected": len(permaslugs),
                        "observed_exact_service_variants_attached": len(exact_ids),
                        "service_variants_minus_permaslugs": len(exact_ids) - len(permaslugs),
                        "service_variant_to_permaslug_ratio": safe_div(len(exact_ids), len(permaslugs)),
                        "distinct_catalog_tokenizer_labels": len(tokenizer_labels),
                        "distinct_catalog_modality_signatures": len(modality_labels),
                        "structured_model_size_records": properties["structured_model_size_available"],
                        "hardware_compatible_weight_design_count": "not_identified",
                    }
                    )

                    if n in policy_ns:
                        selected_summary_rows.append(output)
                        for prop, count in properties.items():
                            prop_token_total = sum(item["total_tokens"] for item in selected if int(item.get(prop) or 0))
                            prop_request_total = sum(item["requests"] for item in selected if int(item.get(prop) or 0))
                            metadata_rows.append(
                            {
                                "period": period,
                                "period_end_date": data["period_end_date"],
                                "universe": universe,
                                "aggregation": aggregation,
                                "rank_basis": basis,
                                "n": n,
                                "available_ranking_units": len(ordered),
                                "metric": prop,
                                "units_with_metric": count,
                                "fraction_selected_units": safe_div(count, n),
                                "metric_token_share_of_total_demand": safe_div(prop_token_total, totals["total_tokens"]),
                                "metric_request_share_of_total_demand": safe_div(prop_request_total, totals["requests"]),
                                "metric_token_share_within_selected": safe_div(prop_token_total, cumulative["total_tokens"]),
                                "metric_request_share_within_selected": safe_div(prop_request_total, cumulative["requests"]),
                            }
                            )
    return coverage_rows, design_rows, metadata_rows, selected_summary_rows


def pearson_rank_correlation(common, rank_a, rank_b):
    if len(common) < 2:
        return ""
    xs = [rank_a[item] for item in common]
    ys = [rank_b[item] for item in common]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return safe_div(numerator, denominator) if denominator else ""


def set_coverage(unit_ids, datasets, period, aggregation, universe):
    units = units_for_universe(datasets[(period, aggregation)]["units"], universe)
    lookup = {row["unit_id"]: row for row in units}
    totals = {measure: sum(row[measure] for row in units) for measure in MEASURES}
    return {
        measure: safe_div(sum(lookup[item][measure] for item in unit_ids if item in lookup), totals[measure])
        for measure in MEASURES
    }


def comparison_outputs(datasets):
    comparisons = []
    specs = []
    for aggregation in AGGREGATIONS:
        for universe in UNIVERSES:
            for basis in RANK_BASES:
                specs.append(("day_vs_week", universe, aggregation, "day", basis, "week", basis))
            for period in PERIODS:
                specs.append(("tokens_vs_requests", universe, aggregation, period, "total_tokens", period, "requests"))

    for comparison_type, universe, aggregation, period_a, basis_a, period_b, basis_b in specs:
        units_a = units_for_universe(datasets[(period_a, aggregation)]["units"], universe)
        units_b = units_for_universe(datasets[(period_b, aggregation)]["units"], universe)
        ordered_a = sorted(units_a, key=lambda row: (-row[basis_a], row["unit_id"]))
        ordered_b = sorted(units_b, key=lambda row: (-row[basis_b], row["unit_id"]))
        rank_a = {row["unit_id"]: rank for rank, row in enumerate(ordered_a, 1)}
        rank_b = {row["unit_id"]: rank for rank, row in enumerate(ordered_b, 1)}
        max_n = min(len(ordered_a), len(ordered_b))
        set_a = set()
        set_b = set()
        for n in range(1, max_n + 1):
            set_a.add(ordered_a[n - 1]["unit_id"])
            set_b.add(ordered_b[n - 1]["unit_id"])
            common = set_a & set_b
            union = set_a | set_b
            row = {
                "comparison_type": comparison_type,
                "universe": universe,
                "aggregation": aggregation,
                "n": n,
                "set_a_period": period_a,
                "set_a_rank_basis": basis_a,
                "set_b_period": period_b,
                "set_b_rank_basis": basis_b,
                "overlap_count": len(common),
                "overlap_fraction_of_n": safe_div(len(common), n),
                "jaccard": safe_div(len(common), len(union)),
                "pearson_correlation_of_global_ranks_in_intersection": pearson_rank_correlation(common, rank_a, rank_b),
            }
            for label, selected_set in (("set_a", set_a), ("set_b", set_b)):
                for evaluation_period in PERIODS:
                    coverage = set_coverage(selected_set, datasets, evaluation_period, aggregation, universe)
                    row["{}_{}_prompt_coverage".format(label, evaluation_period)] = coverage["prompt_tokens"]
                    row["{}_{}_completion_coverage".format(label, evaluation_period)] = coverage["completion_tokens"]
                    row["{}_{}_token_coverage".format(label, evaluation_period)] = coverage["total_tokens"]
                    row["{}_{}_request_coverage".format(label, evaluation_period)] = coverage["requests"]
            comparisons.append(row)
    return comparisons


def check_catalog_size_schema(catalog):
    forbidden_exact_keys = {
        "parameter_count",
        "parameters",
        "active_parameters",
        "num_parameters",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_experts",
        "memory_bytes",
        "weights_bytes",
        "precision",
        "quantization",
    }
    found = set()

    def walk(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                if key.lower() in forbidden_exact_keys:
                    found.add(key)
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    for row in catalog:
        walk(row)
    return sorted(found)


def validate(datasets, coverage_rows, catalog):
    checks = []
    for period in PERIODS:
        exact = datasets[(period, "exact_model_variant")]["units"]
        perma = datasets[(period, "model_permaslug")]["units"]
        for measure in MEASURES:
            exact_total = sum(row[measure] for row in exact)
            perma_total = sum(row[measure] for row in perma)
            if exact_total != perma_total:
                raise AssertionError("{} {} total does not reconcile".format(period, measure))
        checks.append("{} exact/permaslug totals reconcile".format(period))
        for units in (exact, perma):
            if len({row["unit_id"] for row in units}) != len(units):
                raise AssertionError("Duplicate unit IDs")
            if abs(sum(row["token_share"] for row in units) - 1.0) > 1e-12:
                raise AssertionError("Token shares do not sum to one")
            if abs(sum(row["request_share"] for row in units) - 1.0) > 1e-12:
                raise AssertionError("Request shares do not sum to one")
        checks.append("{} unit IDs unique and token/request shares sum to one".format(period))

    grouped = defaultdict(list)
    for row in coverage_rows:
        grouped[(row["period"], row["universe"], row["aggregation"], row["rank_basis"])].append(row)
        for measure in ("prompt", "completion", "token", "request"):
            coverage = row["{}_coverage".format(measure)]
            omitted = row["{}_omitted_share".format(measure)]
            if abs(coverage + omitted - 1.0) > 1e-12:
                raise AssertionError("Coverage/omission complement failed")
            all_coverage = row["{}_coverage_of_all_observed".format(measure)]
            all_omitted = row["{}_omitted_share_of_all_observed".format(measure)]
            if abs(all_coverage + all_omitted - 1.0) > 1e-12:
                raise AssertionError("All-observed coverage/omission complement failed")
    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["n"])
        for field in ("prompt_coverage", "completion_coverage", "token_coverage", "request_coverage"):
            values = [row[field] for row in rows]
            if any(a > b + 1e-15 for a, b in zip(values, values[1:])):
                raise AssertionError("Non-monotone {} for {}".format(field, key))
            if abs(values[-1] - 1.0) > 1e-12:
                raise AssertionError("Final coverage is not one for {} {}".format(key, field))
    checks.append("all top-N coverage series monotone, exhaustive, and complementary to omission")

    found_size_fields = check_catalog_size_schema(catalog)
    if found_size_fields:
        raise AssertionError("Unexpected structured size fields; update analysis: {}".format(found_size_fields))
    checks.append("catalog contains no recognized structured model-size/architecture-size fields")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analyze", "verify"), nargs="?", default="analyze")
    args = parser.parse_args()

    paths, demand, catalog = load_inputs()
    catalog_index = index_catalog(catalog)
    datasets = {}
    for period in PERIODS:
        period_end = demand[period][0]["date"].split()[0]
        for aggregation in AGGREGATIONS:
            datasets[(period, aggregation)] = {
                "units": aggregate_units(demand[period], aggregation, catalog_index),
                "period_end_date": period_end,
                "source_file": str(paths[period].relative_to(ECON_DIR)),
            }

    coverage, design, metadata_rows, selected_summary = topn_outputs(datasets)
    comparisons = comparison_outputs(datasets)
    checks = validate(datasets, coverage, catalog)

    if args.command == "verify":
        print("Internal verification passed ({} checks).".format(len(checks)))
        for check in checks:
            print("- {}".format(check))
        return

    unit_rows, unit_fields = public_unit_rows(datasets)
    ranked_universe_rows, ranked_universe_fields = universe_unit_rows(datasets)
    coverage_fields = list(coverage[0].keys())
    design_fields = list(design[0].keys())
    metadata_fields = list(metadata_rows[0].keys())
    comparison_fields = list(comparisons[0].keys())
    write_csv(HERE / "unit_rankings.csv", unit_rows, unit_fields)
    write_csv(HERE / "ranked_units_by_universe.csv", ranked_universe_rows, ranked_universe_fields)
    write_csv(HERE / "topn_coverage.csv", coverage, coverage_fields)
    write_csv(HERE / "selected_n_summary.csv", selected_summary, coverage_fields)
    write_csv(HERE / "design_count_sensitivity.csv", design, design_fields)
    write_csv(HERE / "metadata_availability.csv", metadata_rows, metadata_fields)
    write_csv(HERE / "selection_stability.csv", comparisons, comparison_fields)

    manifest = {
        "analysis": "static top-N OpenRouter demand coverage; not an estimate of economic inefficiency",
        "inputs": {
            key: {
                "path": str(path.relative_to(ECON_DIR)),
                "sha256": sha256(path),
            }
            for key, path in paths.items()
        },
        "positive_demand_rows": {period: len(demand[period]) for period in PERIODS},
        "output_rows": {
            "unit_rankings.csv": len(unit_rows),
            "ranked_units_by_universe.csv": len(ranked_universe_rows),
            "topn_coverage.csv": len(coverage),
            "selected_n_summary.csv": len(selected_summary),
            "design_count_sensitivity.csv": len(design),
            "metadata_availability.csv": len(metadata_rows),
            "selection_stability.csv": len(comparisons),
        },
        "validation_checks": checks,
    }
    (HERE / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Wrote {} outputs to {}".format(len(manifest["output_rows"]) + 1, HERE))


if __name__ == "__main__":
    main()
