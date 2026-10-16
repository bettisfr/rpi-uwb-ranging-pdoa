"""Rebuild the paper's analysis artifacts in dependency order from a clean checkout."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / 'analysis'
OUTPUT = ANALYSIS / 'outputs'
SRC = ANALYSIS / 'src'

# Each entry lists prerequisites, commands, and required final products.
STAGES = {
    'data': ([], [('main.py', '--build-static-dataset'), ('workflow.py', '--split-static')],
             ['static_grouped_manifest.json', 'static_audit.json']),
    'range-validation': (['data'], [('workflow.py', '--compare-distance')],
                         ['distance_comparison/overall_metrics.csv']),
    'range-lodo': (['data'], [('workflow.py', '--lodo-distance')],
                   ['distance_lodo/overall_metrics.csv']),
    'range-loho': (['data'], [('workflow.py', '--loho-distance')],
                   ['distance_loho/overall_metrics.csv']),
    'phase-lodo': (['data'], [('workflow.py', '--lodo-phase-stratified'),
                             ('workflow.py', '--t0-bearing-label')],
                   ['pdoa_stratified_lodo/overall_metrics.csv', 't0_3_bearing_label/metrics.csv']),
    'phase-calibration': (['data'], [('workflow.py', '--t1-characteristic', '--t0-firmware-frame',
                                    '--pdoa-elevation', 'lodo', '--pdoa-elevation', 'loho',
                                    '--pdoa-elevation', 'lobo'), ('evaluate_phase_tag_loho.py',)],
                          ['pdoa_elevation_lodo/metrics.csv', 'pdoa_elevation_loho/metrics.csv',
                           'pdoa_elevation_lobo/metrics.csv', 'pdoa_tag_diagnostic_loho/metrics.csv']),
    'phase-mlp': (['data'], [('workflow.py', '--lobo-phase-continuous')],
                  ['pdoa_continuous_lobo/overall_metrics.csv']),
    'final-models': (['data'], [('workflow.py', '--final-test'),
                               ('evaluate_range_parametric_test.py',),
                               ('evaluate_position_quadratic_test.py',)],
                     ['final_test/manifest.json', 'range_parametric_test/metrics.csv',
                      'position_quadratic_test/metrics.csv']),
    'static-position': (['range-lodo', 'phase-lodo'], [('workflow.py', '--evaluate-generic-position')],
                        ['position_generic_lodo/overall_metrics.csv']),
    'static-windows': (['range-lodo', 'phase-lodo'],
                       [('windowed_position.py', '--window-size', str(k)) for k in (10, 20, 30)],
                       [f'windowed_position_lodo_w{k}/overall_metrics.csv' for k in (10, 20, 30)]),
    'uav': (['final-models'], [('uav_workflow.py', '--all'), ('evaluate_uav_window_gate.py',)],
             ['uav_range_transfer/metrics.csv', 'uav_pdoa_position/metrics.csv', 'uav_window_gate/metrics.csv']),
    'export': (['range-validation', 'range-lodo', 'range-loho', 'phase-lodo', 'phase-calibration',
                'phase-mlp', 'final-models', 'static-position', 'static-windows', 'uav'],
               [('export_paper_tables.py',)], ['paper_tables/manifest.json']),
}


def ordered_stages(requested):
    ordered = []
    def visit(name):
        for requirement in STAGES[name][0]:
            visit(requirement)
        if name not in ordered:
            ordered.append(name)
    for name in requested:
        visit(name)
    return ordered


def fingerprint():
    digest = hashlib.sha256()
    paths = [*SRC.rglob('*.py'), *(ANALYSIS / 'configs').glob('*.json'),
             *(ROOT / 'data').rglob('*.csv'), *(ROOT / 'data').rglob('*.json'),
             *(ROOT / 'data').rglob('*.kml')]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        with path.open('rb') as stream:
            digest.update(hashlib.file_digest(stream, 'sha256').digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Rebuild every table, including CPU range models and CUDA LOBO MLP')
    parser.add_argument('--stages', choices=tuple(STAGES), nargs='+', help='Selected stages, with prerequisites added automatically')
    parser.add_argument('--resume', action='store_true', help='Reuse completed stages only if code/data fingerprint and products match')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without generating files')
    args = parser.parse_args()
    if not args.all and not args.stages:
        parser.error('choose --all or --stages; use --dry-run to inspect the plan')
    stages = ordered_stages(list(STAGES) if args.all else args.stages)
    if args.dry_run:
        for name in stages:
            for command in STAGES[name][1]:
                print(name + ': ' + ' '.join([sys.executable, str(SRC / command[0]), *command[1:]]))
        return
    if 'phase-mlp' in stages:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit('Full paper reproduction requires CUDA for phase-mlp; install requirements-gpu.txt or choose CPU-only stages.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logs = OUTPUT / 'paper_pipeline_logs'
    logs.mkdir(exist_ok=True)
    ledger_path = OUTPUT / 'paper_pipeline_manifest.json'
    previous = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    current_fingerprint = fingerprint()
    versions = {}
    for package in ('numpy', 'pandas', 'scikit-learn', 'matplotlib', 'torch'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    ledger = dict(fingerprint=current_fingerprint, python=sys.version, versions=versions, stages={})
    if args.resume and previous.get('fingerprint') == current_fingerprint:
        ledger['stages'] = previous.get('stages', {})
    env = dict(os.environ, MPLBACKEND='Agg')
    # Keep linear algebra deterministic and avoid oversubscription; RF retains its configured workers.
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name in stages:
        _, commands, products = STAGES[name]
        if args.resume and ledger['stages'].get(name, {}).get('status') == 'complete' and all((OUTPUT / p).exists() for p in products):
            print(f'{name}: using verified completed stage', flush=True)
            continue
        ledger['stages'][name] = dict(status='running', started=datetime.now(timezone.utc).isoformat())
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
        for number, command in enumerate(commands):
            if command == ('workflow.py', '--final-test'):
                frozen = OUTPUT / 'final_test/manifest.json'
                if frozen.exists() and json.loads(frozen.read_text()).get('status') == 'complete':
                    manifest = json.loads(frozen.read_text())
                    split = json.loads((OUTPUT / 'static_grouped_manifest.json').read_text())
                    if manifest['source_sha256'] != split['source_sha256'] or manifest['split']['seed'] != split['seed']:
                        raise RuntimeError('Existing frozen models do not match the current data/split; use a separate output checkout.')
                    print('final-models: using existing frozen final-test models', flush=True)
                    continue
            log = logs / f'{name}-{number + 1}.log'
            print(f'{name}: {" ".join(command)} -> {log.relative_to(ROOT)}', flush=True)
            with log.open('w') as stream:
                result = subprocess.run([sys.executable, str(SRC / command[0]), *command[1:]],
                                        cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                ledger['stages'][name].update(status='failed', command=list(command), log=str(log.relative_to(ROOT)))
                ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
                raise SystemExit(f'{name} failed: see {log}')
        missing = [p for p in products if not (OUTPUT / p).exists()]
        if missing:
            raise RuntimeError(f'{name} did not create expected products: {missing}')
        ledger['stages'][name].update(status='complete', products=products)
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
    print('Completed requested paper stages.', flush=True)


if __name__ == '__main__':
    main()
