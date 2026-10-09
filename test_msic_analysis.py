"""Offline checks of the paper's model, pinned data, and interactive app.

Run: python -m unittest -v test_msic_analysis
Requires matplotlib and the Streamlit version in requirements.txt.
"""

import contextlib
import csv
import io
import itertools
import json
import math
import random
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import msic_analysis as analysis
import streamlit_app as app


class EconomicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        analysis.verify_inputs()
        cls.D, cls.tokens = analysis.load_demand()

    def test_pinned_universe_and_app_loader(self):
        self.assertEqual(len(self.D), 315)
        self.assertAlmostEqual(math.fsum(self.D.values()), 1, places=9)
        self.assertEqual(app.load_demand_shares(), tuple(sorted(self.D.values(), reverse=True)))

    def test_figure_cells_and_low_baseline_stress(self):
        app.assert_figure_fidelity(tuple(self.D.values()))
        for B in [100e9, *analysis.B_VALUES]:
            for NRE, designs, coverage, savings in analysis.panel_rows(self.D, B):
                result = app.evaluate(tuple(self.D.values()), analysis.S, NRE, B)
                self.assertEqual(result.designs, designs)
                self.assertAlmostEqual(100 * result.coverage, coverage, places=10)
                self.assertAlmostEqual(100 * result.savings, savings, places=10)

    def test_strict_break_even_boundary(self):
        D = {'large': 0.75, 'equal': 0.25, 'zero': 0.0}
        # S=.5, D_M=.25, B=8 gives savings exactly equal to NRE=1.
        with patch.object(analysis, 'S', 0.5):
            self.assertEqual(analysis.msic_set(D, NRE=1, B=8), {'large'})
        result = app.evaluate(tuple(D.values()), S=0.5, NRE=1, B=8)
        self.assertEqual(result.designs, 1)
        self.assertEqual(result.coverage, 0.75)
        self.assertEqual(result.savings, 0.25)

    def test_report_stress_case_is_independent_of_figure_fee_grid(self):
        output = io.StringIO()
        with patch.object(analysis, 'NRE_VALUES', [10e6]), \
             patch.object(analysis, 'make_figure'), \
             contextlib.redirect_stdout(output):
            analysis.main()
        self.assertIn('B=$100B, NRE=$500M stress cell: 26 designs, 86.5% of tokens',
                      output.getvalue())

    def test_figure_single_and_multiple_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            for baselines in ([500e9], analysis.B_VALUES):
                with self.subTest(baselines=baselines):
                    stem = Path(directory) / f"figure-{len(baselines)}"
                    with patch.object(analysis, 'B_VALUES', baselines):
                        analysis.make_figure(self.D, stem=str(stem))
                    for extension in ('.pdf', '.png', '.jpg'):
                        self.assertGreater(stem.with_suffix(extension).stat().st_size, 0)

    def test_app_caption_preserves_savings_rate_precision(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(analysis.HERE / 'streamlit_app.py')).run(timeout=30)
        at.slider[0].set_value(85.4).run()
        self.assertEqual(len(at.exception), 0)
        caption = next(c.value for c in at.caption if c.value.startswith('Current setting:'))
        self.assertIn('S = 85.4%', caption)
        expected = app.evaluate(tuple(self.D.values()), .854, 500e6, 500e9)
        self.assertEqual(at.metric[2].value, f'{100 * expected.savings:.1f}%')

    def test_selection_minimizes_total_resource_cost(self):
        # Independently enumerate every possible allocation on a toy whitelist.
        D = {'alpha': 0.5, 'beta': 0.25, 'gamma': 0.125, 'delta': 0.125}
        for S, NRE, B in itertools.product([0, 0.5, 0.85, 1], [0, 1, 4, 10], [8, 32]):
            def total_cost(chosen):
                return math.fsum(
                    ((1 - S) * D_M * B + NRE) if M in chosen else D_M * B
                    for M, D_M in D.items()
                )

            with patch.object(analysis, 'S', S):
                chosen = analysis.msic_set(D, NRE, B)
            alternatives = (
                total_cost(set(subset))
                for n in range(len(D) + 1)
                for subset in itertools.combinations(D, n)
            )
            self.assertAlmostEqual(total_cost(chosen), min(alternatives))
            outcome = app.evaluate(tuple(D.values()), S, NRE, B)
            self.assertAlmostEqual(outcome.savings, 1 - total_cost(chosen) / B)
            self.assertAlmostEqual(outcome.unserved, 1 - outcome.coverage)
            self.assertGreaterEqual(outcome.savings, -1e-14)

    def test_parameter_sweep_and_scale_invariance(self):
        rng = random.Random(20260905)
        D = tuple(self.D.values())
        for _ in range(100):
            S, NRE, B = rng.uniform(.5, .99), 10 ** rng.uniform(6, 9.3), 10 ** rng.uniform(10.7, 12.7)
            result = app.evaluate(D, S, NRE, B)
            with patch.object(analysis, 'S', S):
                chosen = analysis.msic_set(self.D, NRE, B)
            self.assertEqual(result.designs, len(chosen))
            # Sum each model's positive net savings; do not reuse the coverage formula.
            expected = math.fsum(max(0, S * D_M * B - NRE) for D_M in D) / B
            self.assertAlmostEqual(result.savings, expected, places=12)
            self.assertGreaterEqual(app.evaluate(D, S, NRE / 2, B).coverage, result.coverage)
            self.assertGreaterEqual(app.evaluate(D, S, NRE, B * 2).coverage, result.coverage)
            scaled = app.evaluate(D, S, NRE * 8, B * 8)
            self.assertEqual(scaled.designs, result.designs)
            self.assertAlmostEqual(scaled.savings, result.savings)

    def test_paper_savings_range(self):
        D = tuple(self.D.values())
        endpoints = [app.evaluate(D, S, NRE=500e6, B=500e9) for S in [.6, .95]]
        self.assertEqual([x.designs for x in endpoints], [53, 63])
        self.assertEqual([round(100 * x.coverage, 1) for x in endpoints], [95.2, 96.6])

    def test_price_weighting_claims(self):
        rows = list(analysis.price_weighted_check(self.D, self.tokens).values())
        self.assertEqual([len(grid) for grid in rows], [12, 12])
        self.assertEqual([round(max(100 - row[3] for row in grid), 1) for grid in rows], [2.2, 2.3])
        self.assertEqual([round(max(100 - row[4] for row in grid), 1) for grid in rows], [8.0, 13.8])
        self.assertEqual(round(analysis.free_tier_token_share(), 2), 7.37)
        # Recover token demand directly from raw captures, independently of the CSV.
        raw = analysis.HERE / 'demand_data/openrouter/raw'
        catalog = json.loads(analysis.CATALOG_JSON.read_text())['data']
        whitelist = {r['canonical_slug'] for r in catalog if 'text' in r['architecture']['output_modalities']}
        tokens = defaultdict(int)
        for row in json.loads((raw / 'openrouter_frontend_models_week_20260812.json').read_text())['data']:
            M = row['model_permaslug']
            if M in whitelist:
                tokens[M] += int(row['total_prompt_tokens']) + int(row['total_completion_tokens'])
        self.assertEqual(set(tokens), set(self.D))
        total = sum(tokens.values())
        for M in self.D:
            self.assertAlmostEqual(tokens[M] / total, self.D[M], places=11)

    def test_historical_total_variation(self):
        by_date = defaultdict(dict)
        with analysis.HISTORY_CSV.open() as handle:
            for row in csv.DictReader(handle):
                if row['model_permaslug'] != 'other':
                    self.assertNotIn(row['model_permaslug'], by_date[row['date']])
                    by_date[row['date']][row['model_permaslug']] = float(row['total_tokens'])
        results = analysis.demand_shift_analysis()
        self.assertEqual([row[2] for row in results], [17, 15, 12])
        self.assertEqual([round(row[3]) for row in results], [70, 40, 23])
        for t0, t1, n, shifted in results:
            common = set(by_date[t0]) & set(by_date[t1])
            totals = [math.fsum(by_date[t][M] for M in common) for t in [t0, t1]]
            half_l1 = .5 * math.fsum(abs(by_date[t0][M] / totals[0] - by_date[t1][M] / totals[1]) for M in common)
            self.assertEqual(len(common), n)
            self.assertAlmostEqual(100 * half_l1, shifted)

    def test_app_controls_and_default_render(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(analysis.HERE / 'streamlit_app.py')).run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([m.value for m in at.metric], ['60', '96.25%', '75.8%'])
        self.assertEqual([s.label for s in at.select_slider], ['MSIC development cost NRE', 'Lifetime all-flexible serving cost B'])
        at.slider[0].set_value(60.0).run()
        self.assertEqual(len(at.exception), 0)
        expected = app.evaluate(tuple(self.D.values()), .6, 500e6, 500e9)
        self.assertEqual(at.metric[0].value, str(expected.designs))
        at.select_slider[0].set_value(10_000_000).run()
        at.select_slider[1].set_value(2_000_000_000_000).run()
        self.assertEqual(len(at.exception), 0)
        expected = app.evaluate(tuple(self.D.values()), .6, 10e6, 2e12)
        self.assertEqual(at.metric[0].value, str(expected.designs))


if __name__ == '__main__':
    unittest.main()
