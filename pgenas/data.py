import hashlib
from pathlib import Path
import numpy as np
from pgenas.paper_prompts import SPECS

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_data(folder, branches):
    folder = Path(folder).resolve()
    arrays, files = ({}, {})
    for branch in branches:
        spec = SPECS[branch]
        arrays[branch] = {}
        for role in ('train', 'validation'):
            xp = folder / f'{branch}_{role}_x.npy'
            yp = folder / f'{branch}_{role}_y.npy'
            x = np.load(xp, mmap_mode='r', allow_pickle=False)
            y = np.load(yp, allow_pickle=False)
            if x.ndim != len(spec['shape']) + 1 or list(x.shape[1:]) != spec['shape'] or x.dtype != np.float32 or (len(x) < 2) or (not np.isfinite(x).all()):
                raise ValueError(f'Invalid input array: {xp.name}; expected float32 [N, {spec['shape']}]')
            if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= spec['classes']):
                raise ValueError(f'Invalid label array: {yp.name}')
            arrays[branch].update({role + '_x': str(xp), role + '_y': str(yp), role + '_count': len(x)})
            files[xp.name], files[yp.name] = (sha256(xp), sha256(yp))
    return (arrays, dict(files_sha256=files, test_data_accessed=False, input_stage='preprocessed branch representations'))

def make_demo_data(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for branch, spec in SPECS.items():
        for role, count in (('train', 16), ('validation', 8)):
            np.save(folder / f'{branch}_{role}_x.npy', rng.normal(size=(count, *spec['shape'])).astype(np.float32))
            np.save(folder / f'{branch}_{role}_y.npy', np.arange(count) % spec['classes'])
