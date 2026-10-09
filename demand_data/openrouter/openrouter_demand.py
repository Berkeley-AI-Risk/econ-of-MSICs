#!/usr/bin/env python3
"""Collect and analyze OpenRouter model-demand data.

Primary inputs are official OpenRouter endpoints. The key-authenticated daily
dataset is preferred for history; the public frontend endpoints are useful for
the current cross-section. Historical analysis uses a pinned data-only extract
of OpenRouter rankings from a third-party archive, with recorded provenance.

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"

OFFICIAL_DAY_URL = "https://openrouter.ai/api/frontend/v1/rankings/models?view=day"
OFFICIAL_WEEK_URL = "https://openrouter.ai/api/frontend/v1/rankings/models?view=week"
OFFICIAL_CHART_URL = "https://openrouter.ai/api/frontend/v1/rankings/model-rankings-chart"
OFFICIAL_CATALOG_URL = "https://openrouter.ai/api/v1/models?output_modalities=all"
OFFICIAL_DAILY_URL = "https://openrouter.ai/api/v1/datasets/rankings-daily"
SECONDARY_ARCHIVE_SHA256 = "b613096fcf570d5c8bfbccdc2d4227350e74221cce7627889bf6a3a943c9e91d"
CATALOG_FIELDS = (
    "id", "canonical_slug", "hugging_face_id", "name", "created",
    "context_length", "architecture", "pricing", "expiration_date",
)
CATALOG_NESTED_FIELDS = {
    "architecture": ("input_modalities", "output_modalities", "modality", "tokenizer"),
    "pricing": ("prompt", "completion"),
}


def trim_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields consumed by the analysis and its reproduction tools.

    Preserve row order, values, nulls, and absent fields: these can affect
    identifier matching and the standard-price fallback.
    """
    rows = []
    for source in payload["data"]:
        row = {key: source[key] for key in CATALOG_FIELDS if key in source}
        for key, fields in CATALOG_NESTED_FIELDS.items():
            if isinstance(row.get(key), dict):
                row[key] = {field: row[key][field] for field in fields if field in row[key]}
        rows.append(row)
    return {"data": rows}


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, path: pathlib.Path, token: str | None = None) -> dict[str, Any]:
    headers = {"User-Agent": "sm-asic-demand-research/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    retrieved = utc_timestamp()
    with urllib.request.urlopen(req, timeout=120) as response:
        body = response.read()
        status = response.status
        content_type = response.headers.get("Content-Type")
    record = {
        "url": url,
        "retrieved_at_utc": retrieved,
        "http_status": status,
        "content_type": content_type,
        "file": str(path.relative_to(ROOT)),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    if url == OFFICIAL_CATALOG_URL:
        record["source_sha256"] = record["sha256"]
        record["source_bytes"] = record["bytes"]
        record["transformation"] = "Selected catalog fields consumed by the analysis; values and row order unchanged."
        body = (json.dumps(trim_catalog(json.loads(body)), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        record["bytes"] = len(body)
        record["sha256"] = hashlib.sha256(body).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return record


def collect(args: argparse.Namespace) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    targets = [
        (OFFICIAL_DAY_URL, RAW / f"openrouter_frontend_models_day_{stamp}.json", None),
        (OFFICIAL_WEEK_URL, RAW / f"openrouter_frontend_models_week_{stamp}.json", None),
        (OFFICIAL_CHART_URL, RAW / f"openrouter_frontend_model_rankings_chart_{stamp}.json", None),
        (OFFICIAL_CATALOG_URL, RAW / f"openrouter_models_catalog_{stamp}_trimmed.json", None),
    ]
    if args.daily:
        try:
            end = (dt.date.fromisoformat(args.end_date) if args.end_date else
                   dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1))
            start = dt.date.fromisoformat(args.start_date or "2025-01-01")
        except ValueError as exc:
            raise SystemExit("--start-date and --end-date must be YYYY-MM-DD dates") from exc
        if start > end:
            raise SystemExit("--start-date must be on or before --end-date")
        token = os.environ.get("OPENROUTER_API_KEY")
        if not token:
            raise SystemExit("OPENROUTER_API_KEY is required for --daily")
        # The Data API limits a request to one year. Preserve the requested
        # history in contiguous windows of at most 366 inclusive dates.
        while start <= end:
            window_end = min(end, start + dt.timedelta(days=365))
            query = urllib.parse.urlencode({"start_date": start.isoformat(),
                                           "end_date": window_end.isoformat()})
            filename = f"openrouter_rankings_daily_{start}_{window_end}_{stamp}.json"
            targets.append((f"{OFFICIAL_DAILY_URL}?{query}", RAW / filename, token))
            start = window_end + dt.timedelta(days=1)
    for url, path, token in targets:
        try:
            records.append(fetch(url, path, token))
        except urllib.error.HTTPError as exc:
            records.append({"url": url, "retrieved_at_utc": utc_timestamp(), "http_status": exc.code, "error": str(exc)})
            if url == OFFICIAL_DAILY_URL or "rankings-daily?" in url:
                raise
    manifest = RAW / f"retrieval_manifest_{stamp}.json"
    manifest.write_text(json.dumps({"retrievals": records}, indent=2) + "\n")
    print(manifest)


def latest(pattern: str) -> pathlib.Path:
    matches = sorted(RAW.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching raw/{pattern}")
    return matches[-1]


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def author(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else model


DATE_TAIL = re.compile(r"(?:[-_.](?:19|20)\d{2}(?:[-_.]?\d{2}){0,2})$")
VARIANT_TAIL = re.compile(r":(?:free|beta|extended|thinking|online|batch|nitro)$", re.I)


def family_key(model: str) -> str:
    """Conservative heuristic, not an official OpenRouter family identifier."""
    if model.lower() in {"other", "others"}:
        return "other"
    base = VARIANT_TAIL.sub("", model)
    org, sep, name = base.partition("/")
    name = DATE_TAIL.sub("", name)
    name = re.sub(r"[-_.](?:preview|latest|experimental|exp)$", "", name, flags=re.I)
    return f"{org}/{name}" if sep else name


def concentration(shares: list[float], total_units: int | None = None) -> dict[str, float | None]:
    positive = sorted((x for x in shares if x > 0), reverse=True)
    hhi = sum(x * x for x in positive)
    entropy = -sum(x * math.log(x) for x in positive)
    n = len(positive)
    if n <= 1:
        gini = 0.0
    else:
        asc = sorted(positive)
        gini = (2 * sum((i + 1) * x for i, x in enumerate(asc)) / sum(asc) - (n + 1)) / n
    return {
        "top1_share": sum(positive[:1]),
        "top3_share": sum(positive[:3]),
        "top5_share": sum(positive[:5]),
        "top10_share": sum(positive[:10]),
        "hhi": hhi,
        "effective_count_hhi": 1 / hhi if hhi else None,
        "entropy": entropy,
        "effective_count_entropy": math.exp(entropy),
        "gini_observed_units": gini,
        "observed_units": total_units if total_units is not None else n,
    }


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_public_cross_section(path: pathlib.Path, source_name: str, source_url: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_json(path)
    data = payload["data"]
    period_end = max(str(row["date"])[:10] for row in data)
    out = []
    for row in data:
        prompt = int(row.get("total_prompt_tokens") or 0)
        completion = int(row.get("total_completion_tokens") or 0)
        model = row["model_permaslug"]
        variant = row.get("variant") or "standard"
        exact = row.get("variant_permaslug") or model
        out.append({
            "period_end_date": period_end,
            "last_observed_date": str(row["date"])[:10],
            "model_permaslug": model,
            "variant": variant,
            "exact_model_variant": exact,
            "author": author(model),
            "heuristic_family": family_key(exact),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "requests": int(row.get("count") or 0),
            "reasoning_tokens": int(row.get("total_native_tokens_reasoning") or 0),
            "cached_tokens": int(row.get("total_native_tokens_cached") or 0),
            "tool_calls": int(row.get("total_tool_calls") or 0),
            "source_url": source_url,
            "source_file": str(path.relative_to(ROOT)),
        })
    total_tokens = sum(r["total_tokens"] for r in out)
    total_requests = sum(r["requests"] for r in out)
    for r in out:
        r["token_share"] = r["total_tokens"] / total_tokens if total_tokens else 0
        r["request_share"] = r["requests"] / total_requests if total_requests else 0
    metrics = []
    for measure, denom in (("tokens", total_tokens), ("requests", total_requests)):
        for level, key in (("exact_model_variant", "exact_model_variant"), ("model_permaslug", "model_permaslug"), ("heuristic_family", "heuristic_family"), ("author", "author")):
            vals: dict[str, int] = defaultdict(int)
            field = "total_tokens" if measure == "tokens" else "requests"
            for r in out:
                vals[r[key]] += r[field]
            c = concentration([v / denom for v in vals.values()] if denom else [], len(vals))
            metrics.append({"date": period_end, "source": source_name, "measure": measure, "aggregation": level, "denominator": denom, **c})
    return out, metrics


def parse_archive(path: pathlib.Path) -> dict[str, Any]:
    """Read the verified data-only snapshot; no dashboard code is required."""
    if sha256(path) != SECONDARY_ARCHIVE_SHA256:
        raise ValueError("Secondary archive SHA256 does not match pinned file")
    return load_json(path)


def archive_rows_and_metrics(path: pathlib.Path, selected_dates: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    d = parse_archive(path)
    daily_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    freeze_rows: list[dict[str, Any]] = []
    for idx, date in enumerate(d["dates"]):
        daily = d["daily_models"][idx]
        if daily.get("date") != date:
            raise ValueError("Archive date alignment failure")
        models = daily["models"]
        other = int(daily["other"])
        if date in selected_dates:
            for model, tokens_raw in models.items():
                tokens = int(tokens_raw)
                daily_rows.append({"date": date, "model_permaslug": model, "author": author(model), "heuristic_family": family_key(model), "total_tokens": tokens, "source": "secondary_pinned_archive", "source_file": str(path.relative_to(ROOT))})
            daily_rows.append({"date": date, "model_permaslug": "other", "author": "other", "heuristic_family": "other", "total_tokens": other, "source": "secondary_pinned_archive", "source_file": str(path.relative_to(ROOT))})
        vals = [int(x) for x in models.values()]
        total = int(d["total_tokens"][idx])
        if sum(vals) != int(d["top50_tokens"][idx]) or sum(vals) + other != total:
            raise ValueError(f"Archive total does not reconcile on {date}")
        shares = [x / total for x in vals]
        shares_other_bucket = shares + ([other / total] if other else [])
        c = concentration(shares_other_bucket, len(vals) + (1 if other else 0))
        named_sorted = sorted(shares, reverse=True)
        c.update({
            "top1_share": sum(named_sorted[:1]),
            "top3_share": sum(named_sorted[:3]),
            "top5_share": sum(named_sorted[:5]),
            "top10_share": sum(named_sorted[:10]),
        })
        metrics.append({"date": date, "source": "secondary_pinned_archive", "measure": "tokens", "aggregation": "top50_plus_other_bucket", "denominator": total, "other_share": other / total if total else 0, "hhi_lower_bound": sum(x * x for x in shares), "hhi_upper_bound": sum(x * x for x in shares_other_bucket), **c})

    # Pseudo-freeze retention is observable only for named top-50 rows. The
    # `other` bucket makes these lower bounds on whitelist coverage.
    for cutoff in selected_dates:
        if cutoff not in d["dates"]:
            continue
        ci = d["dates"].index(cutoff)
        whitelist = set(d["daily_models"][ci]["models"])
        for idx in range(ci, len(d["dates"]), 30):
            later = d["daily_models"][idx]["models"]
            known_total = sum(int(x) for x in later.values())
            retained = sum(int(value) for model, value in later.items() if model in whitelist)
            total = int(d["total_tokens"][idx])
            freeze_rows.append({
                "cutoff_date": cutoff,
                "evaluation_date": d["dates"][idx],
                "days_after_cutoff": (dt.date.fromisoformat(d["dates"][idx]) - dt.date.fromisoformat(cutoff)).days,
                "whitelist_named_models": len(whitelist),
                "named_whitelist_share_of_total_lower_bound": retained / total if total else 0,
                "named_whitelist_share_of_visible_top50": retained / known_total if known_total else 0,
                "other_share": int(d["daily_models"][idx]["other"]) / total if total else 0,
                "interpretation": "descriptive lower bound; entrant demand is not a frozen-whitelist counterfactual",
            })
    return daily_rows, metrics, freeze_rows


def chart_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = load_json(path)["data"]
    cached = dt.datetime.fromtimestamp(payload["cachedAt"] / 1000, dt.timezone.utc).isoformat().replace("+00:00", "Z")
    rows = []
    for point in payload["data"]:
        total = sum(int(v) for v in point["ys"].values())
        for model, tokens_raw in point["ys"].items():
            tokens = int(tokens_raw)
            rows.append({"week_start": point["x"], "model_or_others": model, "total_tokens": tokens, "token_share": tokens / total if total else 0, "week_total_tokens": total, "cached_at_utc": cached, "source_url": OFFICIAL_CHART_URL})
    return rows


def cross_validate_archive_weekly(
    archive_path: pathlib.Path, chart_data: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare the secondary daily archive with the first-party weekly chart.

    The weekly chart exposes its leading named models plus ``Others``.  Weekly
    totals can be compared directly.  A named model can be compared only with
    its *visible* daily top-50 tokens in the secondary archive, so its secondary
    value is potentially censored downward when it falls below rank 50 on a day.
    """
    archive = parse_archive(archive_path)
    by_date: dict[str, dict[str, Any]] = {}
    for idx, date in enumerate(archive["dates"]):
        daily = archive["daily_models"][idx]
        by_date[date] = {
            "models": {model: int(tokens) for model, tokens in daily["models"].items()},
            "total": int(archive["total_tokens"][idx]),
        }

    by_week: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chart_data:
        by_week[row["week_start"]].append(row)

    rows: list[dict[str, Any]] = []
    for week_start, official_rows in sorted(by_week.items()):
        start = dt.date.fromisoformat(week_start)
        dates = [(start + dt.timedelta(days=offset)).isoformat() for offset in range(7)]
        if not all(date in by_date for date in dates):
            continue

        official_total = int(official_rows[0]["week_total_tokens"])
        secondary_total = sum(by_date[date]["total"] for date in dates)
        difference = secondary_total - official_total
        rows.append({
            "week_start": week_start,
            "comparison_scope": "weekly_total",
            "model_permaslug": "",
            "official_tokens": official_total,
            "secondary_visible_tokens": secondary_total,
            "signed_difference_tokens": difference,
            "absolute_relative_error": abs(difference) / official_total if official_total else 0,
            "secondary_censoring_note": "none for the archived all-model daily total",
        })

        for official in official_rows:
            model = official["model_or_others"]
            if model == "Others":
                continue
            official_tokens = int(official["total_tokens"])
            secondary_tokens = sum(
                by_date[date]["models"].get(model, 0) for date in dates
            )
            difference = secondary_tokens - official_tokens
            rows.append({
                "week_start": week_start,
                "comparison_scope": "named_model_visible_daily_top50",
                "model_permaslug": model,
                "official_tokens": official_tokens,
                "secondary_visible_tokens": secondary_tokens,
                "signed_difference_tokens": difference,
                "absolute_relative_error": abs(difference) / official_tokens if official_tokens else 0,
                "secondary_censoring_note": "secondary value can omit days below the daily top 50",
            })
    return rows


def catalog_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for item in load_json(path)["data"]:
        arch = item.get("architecture") or {}
        rows.append({
            "id": item.get("id"), "canonical_slug": item.get("canonical_slug"), "name": item.get("name"),
            "created_unix": item.get("created"), "created_utc": dt.datetime.fromtimestamp(item["created"], dt.timezone.utc).isoformat().replace("+00:00", "Z") if item.get("created") else None,
            "context_length": item.get("context_length"), "tokenizer": arch.get("tokenizer"), "modality": arch.get("modality"),
            "input_modalities": "|".join(arch.get("input_modalities") or []), "output_modalities": "|".join(arch.get("output_modalities") or []),
            "expiration_date": item.get("expiration_date"), "source_url": OFFICIAL_CATALOG_URL,
        })
    return rows


def analyze(args: argparse.Namespace) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    day = pathlib.Path(args.day_file) if args.day_file else latest("openrouter_frontend_models_day_*.json")
    week = pathlib.Path(args.week_file) if args.week_file else latest("openrouter_frontend_models_week_*.json")
    archive = pathlib.Path(args.archive_file) if args.archive_file else RAW / "openrouter_rankings_daily_20250101_20260604.json"
    chart = pathlib.Path(args.chart_file) if args.chart_file else latest("openrouter_frontend_model_rankings_chart_*.json")
    catalog = pathlib.Path(args.catalog_file) if args.catalog_file else latest("openrouter_models_catalog_*.json")
    day_rows, day_metrics = parse_public_cross_section(day, "official_public_day", OFFICIAL_DAY_URL)
    week_rows, week_metrics = parse_public_cross_section(week, "official_public_week", OFFICIAL_WEEK_URL)
    write_csv(PROCESSED / "latest_day_model_demand.csv", list(day_rows[0].keys()), sorted(day_rows, key=lambda r: r["total_tokens"], reverse=True))
    write_csv(PROCESSED / "current_week_model_demand.csv", list(week_rows[0].keys()), sorted(week_rows, key=lambda r: r["total_tokens"], reverse=True))
    chart_data = chart_rows(chart)
    write_csv(PROCESSED / "official_weekly_top_models_series.csv", list(chart_data[0].keys()), chart_data)
    catalog_data = catalog_rows(catalog)
    write_csv(PROCESSED / "current_model_catalog.csv", list(catalog_data[0].keys()), catalog_data)
    selected = ["2025-01-01", "2025-06-01", "2026-01-01", "2026-06-04"]
    archive_rows, daily_metrics, freeze_rows = archive_rows_and_metrics(archive, selected)
    write_csv(PROCESSED / "selected_historical_daily_demand.csv", list(archive_rows[0].keys()), archive_rows)
    all_metrics = day_metrics + week_metrics + [m for m in daily_metrics if m["date"] in selected]
    write_csv(PROCESSED / "concentration_summary.csv", list(all_metrics[0].keys()), all_metrics)
    write_csv(PROCESSED / "daily_top50_censoring_summary.csv", list(daily_metrics[0].keys()), daily_metrics)
    write_csv(PROCESSED / "pseudo_freeze_retention.csv", list(freeze_rows[0].keys()), freeze_rows)
    cross_validation = cross_validate_archive_weekly(archive, chart_data)
    write_csv(
        PROCESSED / "secondary_archive_cross_validation.csv",
        list(cross_validation[0].keys()),
        cross_validation,
    )
    print(json.dumps({
        "official_day_file": str(day.relative_to(ROOT)),
        "day_rows": len(day_rows),
        "day_end": max(r["period_end_date"] for r in day_rows),
        "official_week_file": str(week.relative_to(ROOT)),
        "week_rows": len(week_rows),
        "week_end": max(r["period_end_date"] for r in week_rows),
        "archive_days": len(daily_metrics),
        "archive_start": daily_metrics[0]["date"],
        "archive_end": daily_metrics[-1]["date"],
        "output_files": sorted(str(p.relative_to(ROOT)) for p in PROCESSED.glob("*.csv")),
    }, indent=2))


def verify(_: argparse.Namespace) -> None:
    expected = [
        PROCESSED / "current_week_model_demand.csv",
        PROCESSED / "latest_day_model_demand.csv",
        PROCESSED / "official_weekly_top_models_series.csv",
        PROCESSED / "current_model_catalog.csv",
        PROCESSED / "selected_historical_daily_demand.csv",
        PROCESSED / "concentration_summary.csv",
        PROCESSED / "daily_top50_censoring_summary.csv",
        PROCESSED / "pseudo_freeze_retention.csv",
        PROCESSED / "secondary_archive_cross_validation.csv",
    ]
    for path in expected:
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty: {path}")
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SystemExit(f"No data rows: {path}")
        if len(rows) > 1 and len(rows) != len({tuple(r.items()) for r in rows}):
            raise SystemExit(f"Duplicate rows: {path}")
        print(f"OK {path.relative_to(ROOT)}: {len(rows)} rows")

    cross_path = PROCESSED / "secondary_archive_cross_validation.csv"
    with cross_path.open(newline="") as f:
        cross_rows = list(csv.DictReader(f))
    total_errors = [
        float(row["absolute_relative_error"])
        for row in cross_rows
        if row["comparison_scope"] == "weekly_total"
    ]
    named_errors = sorted(
        float(row["absolute_relative_error"])
        for row in cross_rows
        if row["comparison_scope"] == "named_model_visible_daily_top50"
    )
    if len(total_errors) != 41 or len(named_errors) != 369:
        raise SystemExit("Unexpected secondary/archive overlap counts")
    named_p95 = named_errors[math.ceil(0.95 * len(named_errors)) - 1]
    if max(total_errors) > 0.005 or named_p95 > 0.003:
        raise SystemExit("Secondary/archive cross-validation exceeds pinned tolerances")
    print(
        "OK secondary/archive audit: "
        f"{len(total_errors)} weeks, max total error={max(total_errors):.6f}, "
        f"named p95 error={named_p95:.6f}"
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--daily", action="store_true", help="also fetch authenticated top-50 daily data")
    c.add_argument("--start-date")
    c.add_argument("--end-date")
    c.set_defaults(func=collect)
    a = sub.add_parser("analyze")
    a.add_argument("--day-file")
    a.add_argument("--week-file")
    a.add_argument("--archive-file")
    a.add_argument("--chart-file")
    a.add_argument("--catalog-file")
    a.set_defaults(func=analyze)
    v = sub.add_parser("verify")
    v.set_defaults(func=verify)
    return p


if __name__ == "__main__":
    ns = parser().parse_args()
    ns.func(ns)
