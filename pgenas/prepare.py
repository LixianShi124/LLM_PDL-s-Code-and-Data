from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'code'))

from direct_fusion_data import AllDataset, check_disjoint_samples, prepare_cached_dataset
from pgenas.data import sha256
from pgenas.paper_prompts import SPECS


def prepare_search_data(data_root, split, output, cache_root=None):
    source = Path(data_root).resolve() / split
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    inputs = {role: AllDataset(source / (role + '_clean.npz'), role) for role in ('train', 'validation')}
    check_disjoint_samples(inputs['train'], inputs['validation'])
    output.mkdir(parents=True)
    cache_root = Path(cache_root) if cache_root is not None else output / 'cache'
    roles = {}
    for role in ('train', 'validation'):
        source_path = source / (role + '_clean.npz')
        dataset = prepare_cached_dataset(source_path, role, cache_root, 'cpu')
        np.save(output / (role + '_sample_id.npy'), dataset.sample_ids, allow_pickle=False)
        for j, (branch, spec) in enumerate(SPECS.items()):
            x = dataset.features[branch].cpu().numpy()
            y = dataset.labels[:, j].cpu().numpy()
            if x.shape != (len(dataset), *spec['shape']):
                raise ValueError('Unexpected branch feature shape: ' + branch)
            np.save(output / f'{branch}_{role}_x.npy', x, allow_pickle=False)
            np.save(output / f'{branch}_{role}_y.npy', y, allow_pickle=False)
        roles[role] = {'count': len(dataset), 'source_file': str(source_path), 'source_sha256': sha256(source_path)}
    manifest = {'roles': roles, 'split': split, 'test_data_accessed': False,
                'flicker_preprocessing': 'shared_amplitude_envelope_minus_one',
                'transient_preprocessing': 'fundamental_first',
                'files_sha256': {p.name: sha256(p) for p in sorted(output.glob('*.npy'))},
                'preprocessing_sha256': {name: sha256(ROOT / 'code' / name) for name in
                    ('direct_fusion_data.py', 'z1_amp.py', 'z2_harm.py', 'z3_flicker.py', 'z4_tran.py')}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'generated_data')
    parser.add_argument('--split', choices=('split_01', 'split_02', 'split_03'), default='split_01')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path)
    args = parser.parse_args()
    prepare_search_data(args.data_root, args.split, args.output, args.cache_root)


if __name__ == '__main__':
    main()
