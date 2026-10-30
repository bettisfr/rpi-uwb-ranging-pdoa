"""Compare raw and calibrated UAV positions with identical pose and alignment."""
import csv
import json
import numpy as np
from evaluate_uav_pdoa_position import OUTPUT, positions_from_phase
from uav.config import UAV_PREDICTIONS, UAV_REFERENCE, NODE_ANTENNA_OFFSET_M, TAG_HEIGHT_M
from uav.coordinates import q4_heading_deg
from workflow.position_benchmark import write_csv


def main():
    with UAV_PREDICTIONS.open(newline="") as source:
        rows = list(csv.DictReader(source))
    col = lambda key: np.array([float(row[key]) for row in rows])
    flights = np.array([row['flight'] for row in rows])
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    height = col('node_height_m') + NODE_ANTENNA_OFFSET_M - TAG_HEIGHT_M
    rho = np.sqrt(np.maximum(col('raw_range_m') ** 2 - height ** 2, 0))
    heading = q4_heading_deg(json.loads(UAV_REFERENCE.read_text()))
    reports = []
    for flight in sorted(set(flights)):
        train = next(other for other in set(flights) if other != flight)
        mount = manifest['mounts'][f'pdoa_harmonic|{train}']
        x, y, _ = positions_from_phase(rho, height, col('pdoa_deg'), col('yaw_deg'),
                                     heading, mount, col('node_x_m'), col('node_y_m'))
        mask = flights == flight
        error = np.hypot(x - col('tag_x_m'), y - col('tag_y_m'))[mask]
        reports.append(dict(flight=flight, samples=int(mask.sum()), raw_position_mae_m=float(error.mean())))
    write_csv(OUTPUT / 'raw_position_same_alignment.csv', reports)
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
