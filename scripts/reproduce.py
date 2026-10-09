#!/usr/bin/env python3
"""Write the paper's figure, report, and machine-readable grid offline."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import msic_analysis as analysis


def main():
    output = ROOT / 'build'
    output.mkdir(exist_ok=True)
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        analysis.main(figure_stem=str(output / 'econ_fig_1'))
    (output / 'report.txt').write_text(report.getvalue(), encoding='utf-8')
    D, _ = analysis.load_demand()
    rows = [
        dict(S=analysis.S, B=B, NRE=NRE, designs=n,
             coverage_percent=coverage, net_savings_percent=savings)
        for B in analysis.B_VALUES
        for NRE, n, coverage, savings in analysis.panel_rows(D, B)
    ]
    with (output / 'figure-grid.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'figure-grid.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    print(report.getvalue(), end='')
    print(f'Figure, report, CSV, and JSON written to {output}')


if __name__ == '__main__':
    main()
