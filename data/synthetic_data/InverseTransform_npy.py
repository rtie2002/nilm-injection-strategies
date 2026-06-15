import argparse
from pathlib import Path

import numpy as np
import pandas as pd

APPLIANCE_SPECS = {
    'kettle': {'max_power': 3998},
    'microwave': {'max_power': 3969},
    'fridge': {'max_power': 350},
    'dishwasher': {'max_power': 3964},
    'washingmachine': {'max_power': 3999},
}

TIME_COLUMNS = [
    'minute_sin', 'minute_cos',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / 'raw_npy_synthetic_data'
DEFAULT_OUTPUT_DIR = SCRIPT_DIR


def convert_appliance(appliance_name: str, input_dir: Path, output_dir: Path) -> Path:
    max_power = APPLIANCE_SPECS[appliance_name]['max_power']
    npy_path = input_dir / f'ddpm_fake_{appliance_name}_multivariate.npy'
    if not npy_path.exists():
        raise FileNotFoundError(f'Missing input file: {npy_path}')

    generated_npy = np.load(npy_path)
    if generated_npy.ndim != 3 or generated_npy.shape[-1] != 9:
        raise ValueError(f'{npy_path.name}: expected shape (windows, timesteps, 9), got {generated_npy.shape}')

    flat = generated_npy.reshape(-1, 9).copy()
    flat[:, 0] = flat[:, 0] * max_power

    columns = [appliance_name] + TIME_COLUMNS
    out = pd.DataFrame(flat, columns=columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{appliance_name}_synthetic.csv'
    out.to_csv(out_path, index=False)

    print(f'{appliance_name}: saved {out_path.name} | range {out[appliance_name].min():.2f}W to {out[appliance_name].max():.2f}W')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert synthetic multivariate NPY files to CSV with appliance power in watts.'
    )
    parser.add_argument(
        '--appliance',
        type=str,
        default='all',
        choices=['all', 'kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine'],
        help='Appliance to convert, or all available appliances (default: all)',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help='Folder containing raw ddpm_fake_<appliance>_multivariate.npy files',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Folder for converted CSV files',
    )
    args = parser.parse_args()

    print(f'Input dir: {args.input_dir}')
    print(f'Output dir: {args.output_dir}')

    if args.appliance == 'all':
        appliances = []
        for npy_path in sorted(args.input_dir.glob('ddpm_fake_*_multivariate.npy')):
            name = npy_path.name.replace('ddpm_fake_', '').replace('_multivariate.npy', '')
            if name in APPLIANCE_SPECS:
                appliances.append(name)
        if not appliances:
            raise FileNotFoundError(f'No supported NPY files found in {args.input_dir}')
    else:
        appliances = [args.appliance]

    for appliance in appliances:
        convert_appliance(appliance, args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

