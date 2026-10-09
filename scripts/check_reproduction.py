#!/usr/bin/env python3
"""Rebuild archived derived data in scratch space; never overwrite originals."""
import csv
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_csv(original, rebuilt):
    if digest(original) == digest(rebuilt):
        return
    with original.open(newline='', encoding='utf-8') as handle:
        expected = list(csv.reader(handle))
    with rebuilt.open(newline='', encoding='utf-8') as handle:
        actual = list(csv.reader(handle))
    assert len(expected) == len(actual), f'Row count mismatch: {original.name}'
    for index, (left, right) in enumerate(zip(expected, actual)):
        assert len(left) == len(right), f'Column count mismatch: {original.name}, row {index}'
        for column, (a, b) in enumerate(zip(left, right)):
            if a == b:
                continue
            try:
                agrees = math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
            except ValueError:
                agrees = False
            assert agrees, f'{original.name}, row {index}, column {column}: {a!r} != {b!r}'


def main():
    with tempfile.TemporaryDirectory(prefix='msic-rebuild-') as directory:
        scratch = Path(directory).resolve()
        for name in ('demand_data', 'sm_asic_topn'):
            shutil.copytree(ROOT / name, scratch / name, ignore=shutil.ignore_patterns('__pycache__', '.DS_Store'))
        static = scratch / 'sm_asic_topn/static'
        # A newer capture must not change the input chosen for reproduction.
        (scratch / 'demand_data/openrouter/raw/openrouter_frontend_models_week_99999999.json').write_text('{}')
        subprocess.run([sys.executable, str(static / 'analyze_topn.py'), 'analyze'], check=True)
        for original in (ROOT / 'sm_asic_topn/static').glob('*.csv'):
            compare_csv(original, static / original.name)
        assert digest(ROOT / 'sm_asic_topn/static/analysis_manifest.json') == digest(static / 'analysis_manifest.json')
        subprocess.run([sys.executable, str(static / 'verify_outputs.py')], check=True)
        data = scratch / 'demand_data/openrouter'
        raw = data / 'raw'
        subprocess.run([
            sys.executable, str(data / 'openrouter_demand.py'), 'analyze',
            '--day-file', str(raw / 'openrouter_frontend_models_day_20260812.json'),
            '--week-file', str(raw / 'openrouter_frontend_models_week_20260812.json'),
            '--catalog-file', str(raw / 'openrouter_models_catalog_20260812_trimmed.json'),
            '--chart-file', str(raw / 'openrouter_frontend_model_rankings_chart_20260812.json'),
            '--archive-file', str(raw / 'openrouter_rankings_daily_20250101_20260604.json'),
        ], check=True)
        for original in (ROOT / 'demand_data/openrouter/processed').glob('*.csv'):
            compare_csv(original, data / 'processed' / original.name)
        subprocess.run([sys.executable, str(data / 'openrouter_demand.py'), 'verify'], check=True)
    print('PASS: all static and historical derived CSVs reproduce from pinned raw captures (1e-12 floating-point tolerance; nonnumeric fields exact).')


if __name__ == '__main__':
    main()
