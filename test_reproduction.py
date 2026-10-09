"""Packaging regression: rebuilds must use pinned files and reject corruption."""
import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('static_analysis', ROOT / 'sm_asic_topn/static/analyze_topn.py')
static = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static)
spec = importlib.util.spec_from_file_location('demand_analysis', ROOT / 'demand_data/openrouter/openrouter_demand.py')
demand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demand)


class ReproductionTests(unittest.TestCase):
    def test_daily_collection_splits_long_ranges_without_gaps(self):
        args = argparse.Namespace(daily=True, start_date='2025-01-01', end_date='2026-10-08')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(demand, 'RAW', root), \
                 patch.dict(demand.os.environ, {'OPENROUTER_API_KEY': 'test-key'}), \
                 patch.object(demand, 'fetch', return_value={}) as fetch, \
                 contextlib.redirect_stdout(io.StringIO()):
                demand.collect(args)
            daily_calls = [call.args for call in fetch.call_args_list
                           if call.args[0].startswith(demand.OFFICIAL_DAILY_URL + '?')]
            next_start = dt.date(2025, 1, 1)
            paths = set()
            for url, path, token in daily_calls:
                query = parse_qs(urlparse(url).query)
                start = dt.date.fromisoformat(query['start_date'][0])
                end = dt.date.fromisoformat(query['end_date'][0])
                self.assertEqual(start, next_start)
                self.assertGreaterEqual(end, start)
                self.assertLessEqual((end - start).days + 1, 366)
                self.assertNotIn(path, paths)
                paths.add(path)
                self.assertEqual(token, 'test-key')
                next_start = end + dt.timedelta(days=1)
            self.assertEqual(next_start, dt.date(2026, 10, 9))
            self.assertEqual(len(daily_calls), 2)

    def test_daily_collection_rejects_reversed_dates_before_fetching(self):
        args = argparse.Namespace(daily=True, start_date='2026-10-08', end_date='2026-10-01')
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(demand, 'RAW', Path(directory)), \
                 patch.dict(demand.os.environ, {'OPENROUTER_API_KEY': 'test-key'}), \
                 patch.object(demand, 'fetch') as fetch:
                with self.assertRaisesRegex(SystemExit, 'start-date.*end-date'):
                    demand.collect(args)
                fetch.assert_not_called()

    def test_catalog_contains_only_consumed_fields(self):
        paths, _, catalog = static.load_inputs()
        self.assertEqual(len(catalog), 535)
        self.assertEqual(set(demand.load_json(paths['catalog'])), {'data'})
        fields = {'id', 'canonical_slug', 'hugging_face_id', 'name', 'created',
                  'context_length', 'architecture', 'pricing', 'expiration_date'}
        for row in catalog:
            self.assertEqual(set(row), fields)
            self.assertLessEqual(set(row['architecture']),
                                 {'input_modalities', 'output_modalities', 'modality', 'tokenizer'})
            self.assertLessEqual(set(row['pricing']), {'prompt', 'completion'})
        self.assertFalse((demand.RAW / 'openrouter_models_catalog_20260812.json').exists())

    def test_catalog_trim_preserves_order_values_and_missingness(self):
        payload = {'data': [
            {'id': 'z/model:free', 'description': 'omit',
             'architecture': {'tokenizer': None, 'input_modalities': [], 'instruct_type': 'omit'},
             'pricing': {'prompt': '0', 'completion': '-1', 'overrides': ['omit']}},
            {'id': 'a/model', 'architecture': None, 'pricing': None, 'benchmarks': {'omit': 1}},
            {'id': 'a/model', 'pricing': {}},
        ], 'links': {'next': None}, 'total_count': 3}
        before = json.dumps(payload)
        self.assertEqual(demand.trim_catalog(payload), {'data': [
            {'id': 'z/model:free', 'architecture': {'tokenizer': None, 'input_modalities': []},
             'pricing': {'prompt': '0', 'completion': '-1'}},
            {'id': 'a/model', 'architecture': None, 'pricing': None},
            {'id': 'a/model', 'pricing': {}},
        ]})
        self.assertEqual(json.dumps(payload), before)

    def test_catalog_fetch_saves_only_trimmed_data_with_source_hash(self):
        body = b'{"data":[{"id":"test/model","description":"not archived","pricing":{"prompt":"0.1","image":"9"}}]}'
        response = io.BytesIO(body)
        response.status = 200
        response.headers = {'Content-Type': 'application/json'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'raw/catalog_trimmed.json'
            with patch.object(demand, 'ROOT', root), patch.object(demand.urllib.request, 'urlopen', return_value=response):
                record = demand.fetch(demand.OFFICIAL_CATALOG_URL, path)
            self.assertEqual(json.loads(path.read_text()), demand.trim_catalog(json.loads(body)))
            self.assertEqual(record['source_sha256'], hashlib.sha256(body).hexdigest())
            self.assertEqual(record['source_bytes'], len(body))
            self.assertEqual(record['sha256'], demand.sha256(path))
            self.assertEqual(record['bytes'], path.stat().st_size)
            self.assertEqual(record['file'], 'raw/catalog_trimmed.json')

    def test_catalog_provenance_matches_analysis_manifest(self):
        manifest = json.loads((demand.RAW / 'retrieval_manifest_20260812T181706Z.json').read_text())
        paths, _, _ = static.load_inputs()
        derived, = [r for r in manifest['derived_archives'] if r['file'] == paths['catalog'].name]
        original, = [r for r in manifest['retrievals'] if r['file'] == derived['source_file']]
        self.assertEqual(derived['sha256'], demand.sha256(paths['catalog']))
        self.assertEqual(derived['source_sha256'], original['sha256'])
        self.assertFalse(original['included_in_current_tree'])

    def test_historical_snapshot_is_data_only_and_complete(self):
        path = demand.RAW / 'openrouter_rankings_daily_20250101_20260604.json'
        data = demand.parse_archive(path)
        self.assertEqual(set(data), {'provenance', 'dates', 'daily_models', 'total_tokens', 'top50_tokens'})
        self.assertEqual(len(data['dates']), 520)
        self.assertEqual(data['dates'][0], '2025-01-01')
        self.assertEqual(data['dates'][-1], '2026-06-04')
        self.assertEqual(data['dates'], sorted(set(data['dates'])))
        for field in ('daily_models', 'total_tokens', 'top50_tokens'):
            self.assertEqual(len(data[field]), len(data['dates']))
        for date, daily, total, named in zip(data['dates'], data['daily_models'], data['total_tokens'], data['top50_tokens']):
            self.assertEqual(set(daily), {'date', 'models', 'other'})
            self.assertEqual(daily['date'], date)
            self.assertEqual(sum(daily['models'].values()), named)
            self.assertEqual(named + daily['other'], total)
        self.assertFalse(list(demand.RAW.glob('*.html')))

    def test_historical_snapshot_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corrupt.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Secondary archive SHA256'):
                demand.parse_archive(path)

    def test_historical_snapshot_provenance_matches_manifest(self):
        manifest = json.loads((demand.RAW / 'retrieval_manifest_20260812T181706Z.json').read_text())
        derived, = [r for r in manifest['derived_archives'] if r['file'] == 'openrouter_rankings_daily_20250101_20260604.json']
        path = demand.RAW / derived['file']
        self.assertEqual(demand.sha256(path), derived['sha256'])
        provenance = demand.parse_archive(path)['provenance']
        original, = [r for r in manifest['retrievals'] if r['file'] == derived['source_file']]
        self.assertFalse(original['included_in_current_tree'])
        self.assertEqual(original['sha256'], derived['source_sha256'])
        self.assertEqual(original['sha256'], provenance['archive_sha256'])
        self.assertEqual(original['url'], provenance['archive_url'])
        self.assertEqual(original['retrieved_at_utc'], provenance['archive_retrieved_at_utc'])
        self.assertEqual(provenance['license'], 'CC-BY-4.0')

    def test_uses_exact_manifest_captures(self):
        paths, _, _ = static.load_inputs()
        self.assertEqual({p.name for p in paths.values()}, {
            'openrouter_frontend_models_day_20260812.json',
            'openrouter_frontend_models_week_20260812.json',
            'openrouter_models_catalog_20260812_trimmed.json',
        })

    def test_rejects_wrong_manifest_hash(self):
        manifest = json.loads((static.HERE / 'analysis_manifest.json').read_text())
        manifest['inputs']['week']['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary / 'analysis_manifest.json').write_text(json.dumps(manifest))
            with patch.object(static, 'HERE', temporary):
                with self.assertRaisesRegex(ValueError, 'Pinned input hash mismatch'):
                    static.load_inputs()


if __name__ == '__main__':
    unittest.main()
