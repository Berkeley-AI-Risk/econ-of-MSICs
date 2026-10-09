#!/usr/bin/env python3
"""Reproduce the quantitative analysis in the "Economics of MSICs" section
of "The Feasibility of a Hardwired Pause of Frontier AI Training".

The section analyzes the economics of model-specific integrated circuits
(MSICs) for AI inference under a frozen model whitelist.  Notation follows
the text (D[M] in Python is the paper's D_M):

  D_M : model M's share of total inference demand (estimated by its share
        of OpenRouter tokens, week of August 5-11, 2026)
  NRE : fixed cost of developing an MSIC for one model (dollars)
  S   : savings of serving a model on MSICs instead of flexible
        accelerators, as a fraction of flexible serving cost
  B   : expected (discounted) lifetime cost of serving all profitable
        inference demand using only flexible accelerators (dollars)

Assumed decision rule (Question 1): an MSIC is developed for model M iff
S * D_M * B > NRE.

Inputs (all archived in this repository; SHA-256 hashes in the capture
manifests under demand_data/openrouter/raw/):
  - sm_asic_topn/static/ranked_units_by_universe.csv
      per-model token counts and within-universe shares for the
      catalog text-output universe (week ending 2026-08-11)
  - demand_data/openrouter/raw/openrouter_models_catalog_20260812_trimmed.json
      per-model prices, used for the price-weighting robustness check (footnote on D_M estimation)
  - demand_data/openrouter/processed/selected_historical_daily_demand.csv
      archived top-50 rankings at four historical dates, used for the
      demand-shift analysis (footnote in the Changing demand section)

Outputs: econ_fig_1.{pdf,png,jpg} and a printed report of the scenario grid,
price-weighted robustness checks, demand shifts, and replacement illustration.

Requires: Python 3 standard library + matplotlib.
"""

import csv
import hashlib
import json
import math
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
UNIVERSE_CSV = HERE / "sm_asic_topn/static/ranked_units_by_universe.csv"
CATALOG_JSON = HERE / "demand_data/openrouter/raw/openrouter_models_catalog_20260812_trimmed.json"
HISTORY_CSV = HERE / "demand_data/openrouter/processed/selected_historical_daily_demand.csv"
MANIFEST_JSON = HERE / "demand_data/openrouter/raw/retrieval_manifest_20260812T181706Z.json"

N_UNIVERSE = 315          # models with observed traffic in the week of Aug 5-11, 2026
WEEK_END = "2026-08-11"   # period_end_date of every weekly row used below
S = 0.85                  # savings rate used in the adoption figure
NRE_VALUES = [10e6, 50e6, 200e6, 500e6]          # fixed cost per MSIC, dollars
B_VALUES = [500e9, 1e12, 2e12]                 # lifetime all-flexible serving cost, dollars

# The section's numbers are tied to these exact input files.  A hash mismatch
# means the data changed and every downstream number must be re-verified.
EXPECTED_SHA256 = {
    UNIVERSE_CSV: "9c8409dbdff6837aa539a012b314e3421edd93d7af46a9998d91d31828822b97",
    CATALOG_JSON: "6a2403fc2bf1e3fb27acb5663e270ed88aaa4be24cb095103a377f231f2ea54d",
    HISTORY_CSV: "2265b98f22b84488f07100b2beafbd559d1de92eb3b549bc86ac74ac281e525f",
}


def verify_inputs():
    for path, expected in EXPECTED_SHA256.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(
                f"{path.name}: SHA-256 {actual[:16]}... != pinned {expected[:16]}... "
                "(input data changed; re-verify every number in the text)")
    manifest = json.loads(MANIFEST_JSON.read_text())
    records = manifest.get("retrievals", []) + manifest.get("derived_archives", [])
    recorded = {r.get("file"): r.get("sha256") for r in records}
    if recorded.get(CATALOG_JSON.name) != EXPECTED_SHA256[CATALOG_JSON]:
        raise SystemExit("catalog hash disagrees with the capture manifest")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _universe_rows(universe, aggregation):
    """Weekly rows of the requested slice, with date/duplicate/finite guards."""
    rows, seen = [], set()
    with UNIVERSE_CSV.open() as fh:
        for r in csv.DictReader(fh):
            if (r["period"], r["universe"], r["aggregation"]) != (
                    "week", universe, aggregation):
                continue
            assert r["period_end_date"] == WEEK_END, (
                f"unexpected week ending {r['period_end_date']}, pinned {WEEK_END}")
            key = r["unit_id"]
            assert key not in seen, f"duplicate row for {key}"
            seen.add(key)
            for col in ("token_share_within_universe", "prompt_tokens",
                        "completion_tokens"):
                v = float(r[col])
                assert math.isfinite(v) and v >= 0, f"bad {col} for {key}: {v}"
            rows.append(r)
    return rows


def load_demand():
    """Return {model: D_M} (weekly within-universe token shares) and
    {model: (prompt_tokens, completion_tokens)} for the price-weighted check."""
    D, tokens = {}, {}
    for r in _universe_rows("current_catalog_text_output", "model_permaslug"):
        M = r["model_permaslug"]
        D[M] = float(r["token_share_within_universe"])
        tokens[M] = (float(r["prompt_tokens"]), float(r["completion_tokens"]))
    assert abs(sum(D.values()) - 1) < 1e-6, "shares must sum to 1"
    return D, tokens


def exclusion_shares():
    """Share of all observed weekly tokens excluded from the universe:
    non-text-output models and models unmatched to the catalog (the "0.6%
    of tokens" in the D_M footnote)."""
    non_text = unmatched = 0.0
    for r in _universe_rows("all_observed", "model_permaslug"):
        share = float(r["token_share_of_all_observed"])
        if r["catalog_match"] != "1":
            unmatched += share
        elif r["text_output_capable"] != "1":
            non_text += share
    return 100 * non_text, 100 * unmatched


def msic_set(D, NRE, B):
    """Models clearing the paper's assumed threshold: S * D_M * B > NRE."""
    return {M for M, D_M in D.items() if S * D_M * B > NRE}


def panel_rows(D, B):
    """(NRE, n_designs, pct_tokens_on_msics, pct_saving_vs_all_flexible) per fee."""
    rows = []
    for NRE in NRE_VALUES:
        chosen = msic_set(D, NRE, B)
        cov = sum(D[M] for M in chosen)
        saving = S * cov - len(chosen) * NRE / B     # as a fraction of B
        rows.append((NRE, len(chosen), 100 * cov, 100 * saving))
    return rows

# ---------------------------------------------------------------------------
# MSIC adoption figure
# ---------------------------------------------------------------------------

BLUE, ORANGE = "#000000", "#8f8f8f"  # greyscale scheme
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BAR_H = 0.58


def fee_label(NRE):
    return "NRE = \\$" + (f"{NRE/1e6:.0f}M" if NRE < 1e9 else f"{NRE/1e9:g}B")


def make_figure(D, stem="econ_fig_1"):
    fig, axes = plt.subplots(len(B_VALUES), 1, figsize=(6.8, 6.15),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.subplots_adjust(left=0.18, right=0.845, top=0.885, bottom=0.07, hspace=0.55)
    for k, (ax, B) in enumerate(zip(axes, B_VALUES)):
        rows = panel_rows(D, B)
        ys = range(len(rows) - 1, -1, -1)
        for y, (NRE, n, cov, saving) in zip(ys, rows):
            ax.barh(y, cov, height=BAR_H, color=BLUE, edgecolor="white", linewidth=1.5)
            ax.barh(y, 100 - cov, left=cov, height=BAR_H, color=ORANGE,
                    edgecolor="white", linewidth=1.5)
            # match the flexible-remainder's precision so each row's two
            # labels visibly sum to 100
            cov_txt = f"{cov:.2f}" if 100 - cov < 0.1 else f"{cov:.1f}"
            ax.text(1.5, y, f"{n} designs · {cov_txt}% of tokens", color="white",
                    fontsize=8.5, fontweight="bold", va="center")
            rem = 100 - cov
            rem_txt = f"{rem:.2f}% on flexible" if rem < 0.1 else f"{rem:.1f}% on flexible"
            ax.text(min(cov + rem / 2, 98.5), y + BAR_H / 2 + 0.015, rem_txt,
                    color=INK2, fontsize=7.5, ha="right" if cov > 85 else "center",
                    va="bottom")
            ax.text(101.5, y, f"saves {saving:.0f}%", color=INK2, fontsize=8,
                    va="center", clip_on=False)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.55, len(rows) - 0.15)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([fee_label(NRE) for NRE, *_ in rows], fontsize=8.5, color=INK2)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        if k == len(B_VALUES) - 1:
            ax.set_xlabel("Share of observed tokens (out of 315 models on OpenRouter)",
                          color=INK2, fontsize=8.5)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=3)
        ax.tick_params(axis="y", length=0, labelcolor=INK2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title("B = \\$" + (f"{B/1e12:g}T" if B >= 1e12 else f"{B/1e9:.0f}B"),
                     loc="left", fontsize=9, fontweight="normal", color=INK, pad=5)
    axes[0].text(101.5, 3.85, "savings\nvs all-flexible",
                 color=MUTED, fontsize=7.5, va="bottom", clip_on=False)
    fig.legend(handles=[
        Patch(facecolor=BLUE, label="tokens served on dedicated MSICs"),
        Patch(facecolor=ORANGE, label="tokens served on flexible accelerators"),
    ], loc="lower left", bbox_to_anchor=(0.115, 0.925), ncol=2, frameon=False,
       fontsize=8, handlelength=1.2, handleheight=1.0, columnspacing=1.4)
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png", dpi=300)
    fig.savefig(HERE / f"{stem}.jpg", dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Price-weighted robustness check (D_M estimation footnote)
# ---------------------------------------------------------------------------

def _catalog_prices():
    """Two price maps from the pinned catalog: exact listing id (including
    variant suffixes like ':free') -> (prompt, completion) dollars/token,
    and bare model id -> the standard listing's prices."""
    catalog = json.loads(CATALOG_JSON.read_text())
    catalog = catalog.get("data", catalog)
    exact, base = {}, {}
    for model in catalog:
        pr = model.get("pricing") or {}
        try:
            p = (float(pr.get("prompt", 0)), float(pr.get("completion", 0)))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(x) and x >= 0 for x in p):
            continue    # routers use a -1 sentinel for dynamic pricing
        for key in {model.get("canonical_slug"), model.get("id")}:
            if not key:
                continue
            exact.setdefault(key, p)
            stem = key.split(":")[0]
            if stem not in base or base[stem] == (0.0, 0.0):
                base[stem] = p
    return exact, base


def price_weighted_check(D, tokens):
    """Re-estimate D_M by inference spending (tokens x published price) and
    re-run all B panels, under both free-tier conventions reported in the
    D_M footnote:

      A ("standard list price"): each model's total tokens, including its
        free/discounted tiers, are valued at the model's standard price --
        published prices proxy serving cost, which free tokens still incur.
      B ("variant-specific"): each service variant is valued at its own
        posted price, so free-tier tokens carry zero spending weight.

    Under either convention the unserved share of *spending* is ~2%; the
    text reports that number and the free-tier token share (~7%)."""
    exact, base = _catalog_prices()
    conventions = {}

    # Convention A: model-level tokens at the standard list price.
    spend_A, covered_A = {}, {}
    for M, D_M in D.items():
        p = base.get(M)
        if p is None or p == (0.0, 0.0):
            continue
        pt, ct = tokens[M]
        spend_A[M] = pt * p[0] + ct * p[1]
        covered_A[M] = D_M
    conventions["A: all tokens at standard price"] = (spend_A, covered_A)

    # Convention B: each observed variant at its own posted price
    # (free variants without a listing are priced at zero).
    spend_B = defaultdict(float)
    for r in _universe_rows("current_catalog_text_output", "exact_model_variant"):
        M, variant = r["model_permaslug"], r["variant"]
        listing = M if variant == "standard" else f"{M}:{variant}"
        p = exact.get(listing)
        if p is None:
            p = (0.0, 0.0) if r["free_service_variant"] == "1" else base.get(M, (0.0, 0.0))
        spend_B[M] += float(r["prompt_tokens"]) * p[0] + float(r["completion_tokens"]) * p[1]
    conventions["B: variant-specific prices"] = (dict(spend_B), dict(D))

    out = {}
    for label, (spend, covered) in conventions.items():
        total = sum(spend.values())
        D_spend = {M: v / total for M, v in spend.items()}
        rows = []
        for B in B_VALUES:
            for NRE in NRE_VALUES:
                chosen = {M for M, D_M in D_spend.items() if S * D_M * B > NRE}
                rows.append((B, NRE, len(chosen),
                             100 * sum(D_spend[M] for M in chosen),
                             100 * sum(covered.get(M, 0.0) for M in chosen)))
        out[label] = rows
    return out


def free_tier_token_share():
    """Share of universe tokens served on free variants (the "about 7%"
    in the D_M footnote)."""
    return 100 * sum(
        float(r["token_share_within_universe"])
        for r in _universe_rows("current_catalog_text_output", "exact_model_variant")
        if r["free_service_variant"] == "1")


def openrouter_annual_spend():
    """Annualized OpenRouter spending implied by the observed week's tokens
    at standard list prices (the "on the order of $3B per year" in the
    OpenRouter footnote)."""
    D, tokens = load_demand()
    _, base = _catalog_prices()
    weekly = sum(
        tokens[M][0] * base[M][0] + tokens[M][1] * base[M][1]
        for M in D if base.get(M) not in (None, (0.0, 0.0)))
    return weekly * 365.25 / 7

# ---------------------------------------------------------------------------
# Demand shifts within a fixed set of models (Changing demand footnote)
# ---------------------------------------------------------------------------

def demand_shift_analysis():
    """Conditional-probability shifts among fixed sets of models, per the
    methodology of the Changing-demand footnote (subset = models with
    individually reported data at both dates)."""
    by_date = defaultdict(dict)
    with HISTORY_CSV.open() as fh:
        for r in csv.DictReader(fh):
            if r["model_permaslug"].lower() == "other":
                continue
            by_date[r["date"]][r["model_permaslug"]] = float(r["total_tokens"])
    dates = sorted(by_date)
    results = []
    for t0, t1 in zip(dates[:-1], dates[1:]):
        subset = {M for M in by_date[t0] if by_date[t1].get(M, 0) > 0}
        tot0 = sum(by_date[t0][M] for M in subset)
        tot1 = sum(by_date[t1][M] for M in subset)
        p0 = {M: by_date[t0][M] / tot0 for M in subset}
        p1 = {M: by_date[t1][M] / tot1 for M in subset}
        shifted = sum(max(0.0, p1[M] - p0[M]) for M in subset)
        results.append((t0, t1, len(subset), 100 * shifted))
    return results

# ---------------------------------------------------------------------------
# Verification report
# ---------------------------------------------------------------------------

def main(figure_stem="econ_fig_1"):
    verify_inputs()
    print("Inputs verified against pinned SHA-256 hashes.\n")
    D, tokens = load_demand()
    assert len(D) == N_UNIVERSE, f"expected {N_UNIVERSE} models, got {len(D)}"
    print(f"Universe: {N_UNIVERSE} models with observed OpenRouter traffic "
          f"in the week of August 5-11, 2026; S = {S:.0%}")
    non_text, unmatched = exclusion_shares()
    print(f"  excluded from the universe: {non_text:.2f}% of observed tokens "
          f"non-text + {unmatched:.2f}% catalog-unmatched "
          f"= {non_text+unmatched:.1f}%")
    print(f"  OpenRouter annualized spending at standard prices: "
          f"${openrouter_annual_spend()/1e9:.1f}B\n")

    print("[MSIC adoption] Selection under the rule S * D_M * B > NRE")
    worst_unserved, coverages, fee_pcts = 0.0, [], []
    for B in B_VALUES:
        B_label = f"${B/1e12:g}T" if B >= 1e12 else f"${B/1e9:.0f}B"
        for NRE, n, cov, saving in panel_rows(D, B):
            fee_pct = 100 * n * NRE / B
            print(f"  B={B_label:>6}  NRE=${NRE/1e6:>4.0f}M : {n:>3} designs, "
                  f"{cov:6.2f}% of tokens on MSICs, saves {saving:4.1f}% "
                  f"(design fees = {fee_pct:.2f}% of B)")
            worst_unserved = max(worst_unserved, 100 - cov)
            coverages.append(cov)
            fee_pcts.append(fee_pct)
    make_figure(D, stem=figure_stem)
    print(f"  -> {figure_stem}.pdf/.png/.jpg written")

    print(f"\n[Pure-MSIC scenario] Upper bound on the share of tokens unserved\n"
          f"  <= {worst_unserved:.2f}% (least favorable plotted scenario).\n"
          f"  Under the model's assumptions, switching and post-switch MSIC\n"
          f"  creation can only reduce this share. This is a bound, not a prediction.")

    print("\n[Price-weighted demand robustness]")
    print(f"  free-tier variants carry {free_tier_token_share():.2f}% of "
          f"universe tokens")
    for label, rows in price_weighted_check(D, tokens).items():
        worst_tok = max(100 - cov_tok for *_, cov_tok in rows)
        worst_spend = max(100 - cov_spend for *_, cov_spend, _ in rows)
        print(f"  convention {label}: worst-case unserved = "
              f"{worst_spend:.1f}% of spending, {worst_tok:.1f}% of tokens")

    print("\n[Historical demand shifts] Fixed model sets "
          "(archived top-50 rankings)")
    for t0, t1, n, shifted in demand_shift_analysis():
        print(f"  {t0} -> {t1}: subset of {n} models; "
              f"{shifted:.0f}% of within-subset demand shifted "
              f"(total variation distance)")

    print("\n[Selection margins and stress case]")
    # slack multiples at the least favorable cell (B=$500B, NRE=$500M)
    B_w, NRE_w = 500e9, 500e6
    shares_sorted = sorted(D.values(), reverse=True)
    print(f"  slack multiple S*D_M*B/NRE, worst cell: top model = "
          f"{S*shares_sorted[0]*B_w/NRE_w:.0f}x", end="")
    cum = 0.0
    for D_M in shares_sorted:
        cum += D_M
        if cum >= 0.75:
            print(f"; model completing 75% coverage = {S*D_M*B_w/NRE_w:.1f}x")
            break
    # Lower-baseline stress test: B = $100B
    stress_chosen = msic_set(D, NRE=500e6, B=100e9)
    stress_coverage = 100 * sum(D[M] for M in stress_chosen)
    print(f"  B=$100B, NRE=$500M stress cell: {len(stress_chosen)} designs, "
          f"{stress_coverage:.1f}% of tokens on MSICs")

    print(f"\n[Replacement illustration] Fleet-rebuild arithmetic (S = {S:.0%}):")
    print(f"  MSICs serve {min(coverages):.1f}-{max(coverages):.1f}% of tokens "
          f"across the plotted scenarios")
    print(f"  one fleet: (1 - S) = {100*(1-S):.0f}% of B, plus design fees of "
          f"{min(fee_pcts):.2f}-{max(fee_pcts):.1f}% of B per round")
    lo = 2 * (100 * (1 - S)) + 2 * min(fee_pcts)
    hi = 2 * (100 * (1 - S)) + 2 * max(fee_pcts)
    print(f"  full stranding + comparable replacement round: "
          f"2 x serving cost + 2 x design fees = {lo:.1f}-{hi:.1f}% of B")
    print(f"  -> savings vs all-flexible: {100-hi:.0f}-{100-lo:.0f}%")
    print("  Assumes a comparable replacement fleet; not a simulation of demand,\n"
          "  capacity, replacement timing, or discounting.")


if __name__ == "__main__":
    main()
