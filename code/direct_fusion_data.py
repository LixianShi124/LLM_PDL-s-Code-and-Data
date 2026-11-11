from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from z1_amp import AmplitudeSignalProcessor
from z2_harm import HarmonicSignalProcessor
from z3_flicker import FlickerSignalProcessor
from z4_tran import TranSignalProcessor
from confidence_calibration import to_paper_labels, map_branch_predictions, PAPER_LABEL_NAMES
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / 'generated_data' / 'split_01'
TRAIN_FILE = DATA_DIR / 'train_clean.npz'
VALIDATION_FILE = DATA_DIR / 'validation_clean.npz'
MODEL_DIR = HERE / 'Model'

class AllDataset(Dataset):

    def __init__(self, file_path, mode='train'):
        if mode not in ('train', 'validation', 'test'):
            raise ValueError('mode must be train, validation or test')
        path = Path(file_path).resolve()
        protocol_path = path.parent.parent / 'protocol.json'
        if protocol_path.is_file():
            protocol = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
            if protocol.get('schema_version') != 'pqd-revision-clean-v2':
                raise ValueError('The dataset protocol must be corrected V2')
        if mode != 'test' and path.name.lower().startswith('test'):
            raise ValueError('Test data cannot be used for training or validation calibration')
        self.mode = mode
        self.file_paths = [str(path)]
        self.label_order = PAPER_LABEL_NAMES
        if path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as arrays:
                self.data = arrays['X'].astype(np.float32)
                storage_y = arrays['y'].copy()
                branches = arrays['branch_labels'].copy()
                self.sample_ids = arrays['sample_id'].copy() if 'sample_id' in arrays else None
                self.class_ids = arrays['class_id'].astype(np.int64) if 'class_id' in arrays else np.full(len(self.data), -1)
        elif path.suffix.lower() == '.csv':
            raw = pd.read_csv(path)
            self.sample_ids = raw.pop('sample_id').to_numpy(dtype=str) if 'sample_id' in raw.columns else None
            if raw.shape[1] < 1213:
                raise ValueError('CSV requires 1200 voltage columns, 8 labels, 4 branch labels and final ID')
            self.data = raw.iloc[:, :1200].to_numpy(dtype=np.float32)
            storage_y = raw.iloc[:, -13:-5].to_numpy()
            branches = raw.iloc[:, -5:-1].to_numpy()
            self.class_ids = raw.iloc[:, -1].to_numpy(dtype=np.int64)
        else:
            raise ValueError('Expected an existing CSV or NPZ dataset')
        self.num_samples = len(self.data)
        if self.data.shape != (self.num_samples, 1200) or not np.isfinite(self.data).all():
            raise ValueError('Voltage data must be finite with shape (N,1200)')
        if branches.shape != (self.num_samples, 4) or storage_y.shape != (self.num_samples, 8):
            raise ValueError('Unexpected branch or multilabel shape')
        if not np.isin(storage_y, (0, 1)).all():
            raise ValueError('Multilabel targets must be binary')
        for j, count in enumerate((4, 2, 2, 4)):
            if not np.isin(branches[:, j], np.arange(count)).all():
                raise ValueError('Invalid branch labels')
        self.Multi_labels = to_paper_labels(storage_y).astype(np.int64)
        expected = map_branch_predictions(*(torch.as_tensor(branches[:, j]) for j in range(4))).numpy()
        if not np.array_equal(expected, self.Multi_labels):
            raise ValueError('Branch labels and multilabel targets are inconsistent')
        self.Amp_labels, self.Harm_labels, self.Flicker_labels, self.Tran_labels = branches.astype(np.int64).T
        if self.sample_ids is None:
            self.sample_ids = np.array([f'{path}:{i}' for i in range(self.num_samples)])
        if len(self.sample_ids) != self.num_samples or len(set(self.sample_ids.tolist())) != self.num_samples:
            raise ValueError('Sample IDs must be unique within a dataset')
        self.Amp_processor = AmplitudeSignalProcessor(fs=6000.0)
        self.Harm_processor = HarmonicSignalProcessor(fs=6000.0)
        self.Flicker_processor = FlickerSignalProcessor(fs=6000.0)
        self.Tran_processor = TranSignalProcessor(fs=6000.0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        raw = self.data[idx]
        amplitude = self.Amp_processor.get_amplitude_channel(raw)
        features = (amplitude, self.Harm_processor.get_harmonic_channel(raw), self.Flicker_processor.get_flicker_channel(amplitude[1])[None, :], self.Tran_processor.get_transient_channel(raw, fundamental=amplitude[0]))
        labels = (self.Amp_labels[idx], self.Harm_labels[idx], self.Flicker_labels[idx], self.Tran_labels[idx])
        result = []
        for feature, label in zip(features, labels):
            result.extend([torch.tensor(np.asarray(feature).copy(), dtype=torch.float32), torch.tensor(label, dtype=torch.long)])
        return (*result, torch.tensor(self.Multi_labels[idx], dtype=torch.long))

class CombinedTestDataset(ConcatDataset):

    def __init__(self, datasets):
        super().__init__(datasets)
        self.mode = 'test'
        self.Multi_labels = np.concatenate([d.Multi_labels for d in datasets])
        self.sample_ids = np.concatenate([d.sample_ids for d in datasets])
        self.class_ids = np.concatenate([d.class_ids for d in datasets])
        self.file_paths = [p for d in datasets for p in d.file_paths]
        self.label_order = PAPER_LABEL_NAMES
        if len(set(self.sample_ids.tolist())) != len(self.sample_ids):
            raise ValueError('Sample IDs overlap between seen and unseen test sets')

def make_test_dataset(data_dir=DATA_DIR, mode='test'):
    if mode != 'test':
        raise ValueError('make_test_dataset only loads test data')
    return CombinedTestDataset([AllDataset(Path(data_dir) / f'{role}_clean.npz', mode='test') for role in ('test_seen', 'test_unseen')])

def check_disjoint_samples(*datasets):
    used = set()
    for dataset in datasets:
        ids = set(dataset.sample_ids.tolist())
        if used.intersection(ids):
            raise ValueError('Sample IDs overlap between training, validation or test data')
        used.update(ids)

def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

class CachedBranchDataset(Dataset):

    def __init__(self, manifest_path, device='cpu'):
        self.manifest_path = Path(manifest_path).resolve()
        self.cache_provenance = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        info = self.cache_provenance
        self.mode, self.file_paths = (info['mode'], [info['source_path']])
        self.label_order = PAPER_LABEL_NAMES
        directory = self.manifest_path.parent
        for name, expected in info['files_sha256'].items():
            if _sha256_file(directory / name) != expected:
                raise ValueError('Cached feature checksum mismatch: ' + name)
        self.features = {branch: torch.from_numpy(np.load(directory / f'{branch}.npy', mmap_mode='c')) for branch in ('amp', 'harm', 'flicker', 'tran')}
        self.labels = torch.from_numpy(np.load(directory / 'branch_labels.npy'))
        self.Multi_labels = np.load(directory / 'paper_labels.npy')
        self.sample_ids = np.load(directory / 'sample_ids.npy', allow_pickle=False)
        self.class_ids = np.load(directory / 'class_ids.npy')
        self.to(device)

    def to(self, device):
        self.features = {branch: values.to(device) for branch, values in self.features.items()}
        self.labels = self.labels.to(device)
        return self

    def __len__(self):
        return len(self.Multi_labels)

    def __getitem__(self, index):
        result = []
        for j, branch in enumerate(('amp', 'harm', 'flicker', 'tran')):
            result.extend([self.features[branch][index], self.labels[index, j]])
        return (*result, torch.as_tensor(self.Multi_labels[index], dtype=torch.long))

def prepare_cached_dataset(file_path, mode, cache_root, device='cpu'):
    path = Path(file_path).resolve()
    if mode not in ('train', 'validation', 'test'):
        raise ValueError('Invalid dataset mode')
    if mode != 'test' and path.name.lower().startswith('test'):
        raise ValueError('Test data cannot be used for training or validation calibration')
    descriptor = {'schema': 'direct-branch-cache-v1', 'source_path': str(path), 'source_sha256': _sha256_file(path), 'mode': mode, 'preprocessing_sha256': {name: _sha256_file(HERE / name) for name in ('z1_amp.py', 'z2_harm.py', 'z3_flicker.py', 'z4_tran.py', 'direct_fusion_data.py')}, 'label_order': list(PAPER_LABEL_NAMES)}
    cache_id = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(cache_root) / path.parent.name / mode / cache_id
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        return CachedBranchDataset(manifest_path, device)
    source = AllDataset(path, mode)
    if not len(source):
        raise ValueError('Cannot cache an empty dataset')
    directory.mkdir(parents=True, exist_ok=True)
    first = source[0]
    features = {branch: np.lib.format.open_memmap(directory / f'{branch}.npy', mode='w+', dtype=np.float32, shape=(len(source), *first[2 * j].shape)) for j, branch in enumerate(('amp', 'harm', 'flicker', 'tran'))}
    for i in range(len(source)):
        item = first if i == 0 else source[i]
        for j, branch in enumerate(features):
            features[branch][i] = item[2 * j].numpy()
        if (i + 1) % 2000 == 0 or i + 1 == len(source):
            pass
    for values in features.values():
        values.flush()
    del features
    np.save(directory / 'branch_labels.npy', np.column_stack((source.Amp_labels, source.Harm_labels, source.Flicker_labels, source.Tran_labels)))
    np.save(directory / 'paper_labels.npy', source.Multi_labels)
    np.save(directory / 'sample_ids.npy', source.sample_ids, allow_pickle=False)
    np.save(directory / 'class_ids.npy', source.class_ids)
    descriptor['n_samples'] = len(source)
    descriptor['files_sha256'] = {p.name: _sha256_file(p) for p in sorted(directory.glob('*.npy'))}
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(manifest_path)
    return CachedBranchDataset(manifest_path, device)
