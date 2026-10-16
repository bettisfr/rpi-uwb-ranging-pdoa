"""Export numeric paper tables with source provenance; never fit a model."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from uav.config import TAG_LAYOUT, UAV_DATA
from workflow.phase import circular_summary

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / 'analysis'
OUTPUT = ANALYSIS / 'outputs'
DESTINATION = OUTPUT / 'paper_tables'
SOURCES = {}


def read(relative):
    path = ANALYSIS / relative
    if not path.exists():
        raise FileNotFoundError(f'Missing {relative}; run reproduce_paper.py --all (or its relevant stage).')
    with path.open('rb') as stream:
        SOURCES[str(path.relative_to(ROOT))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    if not rows:
        raise ValueError(f'No rows for {name}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (DESTINATION / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(row, protocol, source, unit, position=False):
    prefix = 'position_' if position else ''
    suffix = 'deg' if unit == 'deg' else 'm'
    scale = 100. if unit == 'cm' else 1.
    result = dict(protocol=protocol, model=row['method'], unit=unit,
                  samples=int(row['samples']), source_csv=source)
    for metric in ('mae', 'rmse', 'p90', 'mae_ci_low', 'mae_ci_high'):
        key = f'{prefix}{metric}_{suffix}'
        if key in row and row[key] != '':
            result[metric] = scale * float(row[key])
    for field in ('coverage', 'accepted_fraction'):
        if field in row:
            result['coverage_percent'] = 100. * float(row[field])
    return result


def static_description():
    rows = read('csv/static_measurements.csv')
    series, conditions = defaultdict(list), defaultdict(list)
    for row in rows:
        series[(row['experiment_id'], row['run_file'], row['tag'])].append(row)
        conditions[(row['height_difference_m'], row['target_ground_distance_m'],
                    row['tag_rotation_deg'], row['tag'])].append(row)
    coverage = []
    for height in sorted({r['height_difference_m'] for r in rows}, key=float):
        runs = defaultdict(dict)
        for (experiment, run, tag), records in series.items():
            if records[0]['height_difference_m'] == height:
                runs[(experiment, run)][tag] = len(records)
        complete = sum(len(tags) == 9 and all(n == 200 for n in tags.values()) for tags in runs.values())
        coverage.append(dict(height_m=height, runs=len(runs), complete=complete,
                             partial=len(runs) - complete, records=sum(sum(t.values()) for t in runs.values())))
    write('static_dataset', coverage)
    tags = []
    for tag in sorted({r['tag'] for r in rows}):
        tag_series = [records for key, records in series.items() if key[2] == tag]
        tag_conditions = [records for key, records in conditions.items() if key[3] == tag]
        samples = [r for records in tag_series for r in records]
        seconds = sum((max(datetime.fromisoformat(r['time']) for r in records) -
                       min(datetime.fromisoformat(r['time']) for r in records)).total_seconds()
                      for records in tag_series)
        tags.append(dict(tag=tag, runs=len(tag_series), records=len(samples), acquisition_seconds=seconds,
            mean_within_condition_range_std_cm=100. * float(np.mean([
                np.std([float(r['range_m']) for r in records], ddof=1)
                for records in tag_conditions if len(records) > 1])),
            mean_within_condition_phase_std_deg=float(np.mean([
                circular_summary([float(r['pdoa_deg']) for r in records])['circular_std_deg']
                for records in tag_conditions if len(records) > 1])),
            circular_mean_phase_deg=circular_summary([float(r['pdoa_deg']) for r in samples])['circular_bias_deg']))
    write('static_tag_summary', tags)


def uav_description():
    logs = sorted(UAV_DATA.glob('drone_*.csv'))
    flights, tag_counts = [], defaultdict(int)
    for number, (log, kml) in enumerate(zip(logs, sorted((UAV_DATA / 'dji-flight-records').glob('*.kml'))), 1):
        with log.open(newline='') as stream:
            aligned = list(csv.DictReader(stream))
        for row in aligned:
            tag_counts[row['tag']] += 1
        coordinates = ET.parse(kml).find('.//{http://www.opengis.net/kml/2.2}coordinates')
        points = [tuple(map(float, value.split(',')[:2])) for value in coordinates.text.split()]
        distance = 0.
        for (lon1, lat1), (lon2, lat2) in zip(points, points[1:]):
            a, b = math.radians(lat1), math.radians(lat2)
            value = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(math.radians(lon2-lon1)/2)**2
            distance += 6371000. * 2 * math.asin(min(1., math.sqrt(value)))
        with kml.with_suffix('.csv').open(newline='') as stream:
            telemetry = next(csv.DictReader(stream))
        duration = float(telemetry['DETAILS.totalTime'])
        flights.append(dict(flight=number, flight_id=log.stem, duration_s=duration,
                            path_km=distance / 1000., mean_speed_mps=distance / duration,
                            max_speed_mps=float(telemetry['DETAILS.maxHorizontalSpeed']),
                            mean_fc_height_m=float(np.mean([float(r['height_fusion_m']) for r in aligned])),
                            records=len(aligned)))
    write('uav_flight_summary', flights)
    write('uav_tag_layout', [dict(tag=tag, bearing_deg=bearing, distance_m=distance,
          rotation_deg=(index % 4)*90, records=tag_counts[tag])
          for index, (tag, (bearing, distance)) in enumerate(TAG_LAYOUT.items())])


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / 'manifest.json').write_text(json.dumps(dict(status='running')) + '\n')
    static_description()
    uav_description()
    ranges = []
    for protocol, directory in (('grouped validation', 'distance_comparison'), ('LODO', 'distance_lodo'), ('LOHO', 'distance_loho')):
        source = f'outputs/{directory}/overall_metrics.csv'
        ranges += [metric_row(r, protocol, source, 'cm') for r in read(source)]
    source = 'outputs/range_parametric_test/metrics.csv'
    ranges += [metric_row(r, 'Test', source, 'cm') for r in read(source) if r['height_difference_m'] == 'all']
    write('range_models', ranges)

    phase = []
    for protocol in ('lodo', 'loho', 'lobo'):
        source = f'outputs/pdoa_elevation_{protocol}/metrics.csv'
        phase += [metric_row(r, protocol.upper(), source, 'deg') for r in read(source) if r['stratum'] == 'all']
    source = 'outputs/pdoa_continuous_lobo/overall_metrics.csv'
    phase += [metric_row(r, 'LOBO', source, 'deg') for r in read(source) if r['method'].startswith('mlp')]
    source = 'outputs/pdoa_stratified_lodo/overall_metrics.csv'
    phase += [metric_row(r, 'LODO diagnostic', source, 'deg') for r in read(source) if r['method'] == 'trigonometric_bearing']
    source = 'outputs/t0_3_bearing_label/metrics.csv'
    phase += [metric_row(r, 'LODO diagnostic', source, 'deg') for r in read(source)
              if r['height_difference_m'] == 'all' and r['method'] == 'label_only_constant']
    source = 'outputs/pdoa_tag_diagnostic_loho/metrics.csv'
    phase += [metric_row(r, 'LOHO diagnostic', source, 'deg') for r in read(source)
              if r['held_height_m'] == 'all' and r['method'] != 'raw']
    source = 'outputs/final_test/pdoa_metrics.csv'
    phase += [metric_row(r, 'Test', source, 'deg') for r in read(source)
              if r['height_difference_m'] == 'all' and r['scope'] == 'all']
    write('pdoa_models', phase)

    source = 'outputs/position_generic_lodo/overall_metrics.csv'
    positions = [metric_row(r, 'LODO', source, 'm', position=True) for r in read(source)]
    source = 'outputs/position_quadratic_test/metrics.csv'
    positions += [metric_row(r, f"Test {r['scope']}", source, 'm', position=True) for r in read(source)
                  if r['height_difference_m'] == 'all' and r['method'] == 'quadratic_regression_global_harmonic']
    write('position_models', positions)
    windows = []
    for k in (10, 20, 30):
        source = f'outputs/windowed_position_lodo_w{k}/overall_metrics.csv'
        for row in read(source):
            if row['method'] != f'geometry_joint_window{k}':
                continue
            record = metric_row(row, 'LODO development windows', source, 'm', position=True)
            record.update(window_size=k, selection=row['gate'])
            windows.append(record)
    write('window_position', windows)

    source = 'outputs/uav_range_transfer/metrics.csv'
    range_rows = read(source)
    flight_ids = sorted({r['group'] for r in range_rows if r['group'].startswith('drone_')})
    position_source = 'outputs/uav_pdoa_position/metrics.csv'
    position_rows = read(position_source)
    window_source = 'outputs/uav_window_gate/metrics.csv'
    window_rows = read(window_source)
    uav = []
    for number, flight in enumerate(flight_ids, 1):
        other = next(f for f in flight_ids if f != flight)
        single = next(r for r in position_rows if r['train_flight'] == other and r['scope'] == 'all' and r['method'] == 'pdoa_harmonic')
        full = next(r for r in window_rows if r['train_flight'] == other and r['window_size'] == '20' and r['gate'] == 'all')
        selected = next(r for r in window_rows if r['train_flight'] == other and r['window_size'] == '20' and r['gate'] == 'std<=20')
        raw_range = next(r for r in range_rows if r['group'] == flight and r['method'] == 'raw_horizontal')
        corrected_range = next(r for r in range_rows if r['group'] == flight and r['method'] == 'static_quadratic_horizontal')
        uav.append(dict(flight=number, flight_id=flight, records=single['samples'],
            range_raw_mae_m=raw_range['mae_m'], range_corrected_mae_m=corrected_range['mae_m'],
            single_mae_m=single['position_mae_m'], window_mae_m=full['position_mae_m'],
            same_windows_single_mae_m=full['same_windows_single_sample_mae_m'],
            selected_window_mae_m=selected['position_mae_m'],
            selected_same_windows_single_mae_m=selected['same_windows_single_sample_mae_m'],
            selected_ci_low_m=selected['ci_low_m'], selected_ci_high_m=selected['ci_high_m'],
            coverage_percent=100.*float(selected['coverage']), windows=full['windows'], selected_windows=selected['windows'],
            range_source_csv=source, position_source_csv=position_source, window_source_csv=window_source))
    write('uav_transfer', uav)
    manifest = dict(status='complete', source_files_sha256=SOURCES,
                    primary_position_chain='quadratic horizontal range + global harmonic phase, static and UAV',
                    tables=[p.name for p in sorted(DESTINATION.glob('*.csv'))],
                    note='Exports contain computed results without manual replacement; preserve full precision until rendering.')
    (DESTINATION / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(manifest["tables"])} numeric tables to {DESTINATION}')


if __name__ == '__main__':
    main()
