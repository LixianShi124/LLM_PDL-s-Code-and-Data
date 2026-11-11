from pathlib import Path
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from direct_fusion_data import AllDataset, make_test_dataset, check_disjoint_samples, prepare_cached_dataset, DATA_DIR, MODEL_DIR
from confidence_calibration import BRANCH_NAMES, PAPER_LABEL_NAMES, map_branch_predictions, calibrate_branches, calibrated_predictions, PredictionRecorder
from mdgr import compute_mdgr_loss
from architectures.amp.architecture import Amplitude_Net_Opt
from architectures.harm.architecture import Harmonic_Net_Opt
from architectures.flicker.architecture import Flicker_Net_Opt
from architectures.tran.architecture import Transient_Net_Opt
HERE = Path(__file__).resolve().parent
seed = 42

def map_preds_to_vector(p_amp, p_harm, p_flicker, p_tran):
    return map_branch_predictions(p_amp, p_harm, p_flicker, p_tran)

def _training_file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _training_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def _atomic_training_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)

def _training_initial_hash(net):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def _training_dataset_evidence(dataset):
    import hashlib
    evidence = {'mode': dataset.mode, 'n': len(dataset), 'sample_id_sha256': hashlib.sha256(json.dumps(dataset.sample_ids.tolist()).encode()).hexdigest(), 'source_sha256': {str(p): _training_file_hash(p) for p in getattr(dataset, 'file_paths', []) if Path(p).is_file()}}
    manifest = getattr(dataset, 'manifest_path', None)
    if manifest is not None:
        evidence['cache_manifest_path'] = str(Path(manifest).resolve())
        evidence['cache_manifest_sha256'] = _training_file_hash(manifest)
    evidence['cache_provenance'] = getattr(dataset, 'cache_provenance', {})
    return evidence

def load_training_models(architecture_dir, device):
    if architecture_dir is None:
        constructors = (Amplitude_Net_Opt, Harmonic_Net_Opt, Flicker_Net_Opt, Transient_Net_Opt)
        return {b: cls(num_classes=c).to(device) for b, cls, c in zip(BRANCH_NAMES, constructors, (4, 2, 2, 4))}
    import importlib.util
    classes = ('Amplitude_Net_Opt', 'Harmonic_Net_Opt', 'Flicker_Net_Opt', 'Transient_Net_Opt')
    result = {}
    public_classes = ('AmpNet', 'HarmNet', 'FlickNet', 'TranNet')
    for branch, name, public_name, count in zip(BRANCH_NAMES, classes, public_classes, (4, 2, 2, 4)):
        path = Path(architecture_dir) / branch / 'architecture.py'
        spec = importlib.util.spec_from_file_location('fixed_' + branch + '_' + _training_file_hash(path)[:12], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        constructor = getattr(module, name, None) or getattr(module, public_name, None)
        if constructor is None:
            raise ValueError('Missing supported branch network class: ' + branch)
        result[branch] = constructor(num_classes=count).to(device)
    return result

def _training_logits(value, count, classes):
    value = value[0] if isinstance(value, (tuple, list)) else value
    if not isinstance(value, torch.Tensor) or value.shape != (count, classes) or (not torch.isfinite(value).all()):
        raise ValueError('Invalid/nonfinite training logits')
    return value

def _training_batches(dataset, order, batch_size, device):
    if hasattr(dataset, 'features') and hasattr(dataset, 'labels'):
        for start in range(0, len(order) - batch_size + 1, batch_size):
            indices = order[start:start + batch_size]
            inputs = [dataset.features[b].index_select(0, torch.as_tensor(indices, device=dataset.features[b].device)).to(device) for b in BRANCH_NAMES]
            targets = dataset.labels.index_select(0, torch.as_tensor(indices, device=dataset.labels.device)).to(device)
            yield (inputs, [targets[:, j] for j in range(4)])
    else:
        loader = DataLoader(dataset, batch_size=batch_size, sampler=order.tolist(), drop_last=True)
        for batch in loader:
            yield ([batch[2 * i].to(device) for i in range(4)], [batch[2 * i + 1].to(device) for i in range(4)])

def _verify_training_record(model_dir, config):
    record_path = Path(model_dir) / 'training_record.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record.get('config') != config or record.get('state') != 'calibrated':
        raise ValueError('Completed training config/source changed')
    for path, digest in record['frozen_files'].items():
        if _training_file_hash(path) != digest:
            raise ValueError('Frozen weights/temperature/training evidence changed')
    return record

def train_independent_decoupled(data_dir=DATA_DIR, train_file=None, validation_file=None, model_dir=MODEL_DIR, output_dir=None, epochs=100, batch_size=128, irm_base_weight=0.5, device=None, training_seed=None, train_dataset=None, validation_dataset=None, architecture_dir=None, run_test=True, resume=False, calibration_filename=None):
    import hashlib
    import time
    from confidence_calibration import load_temperatures
    if epochs < 1 or batch_size < 1 or irm_base_weight < 0:
        raise ValueError('epochs/batch_size must be positive; IRM weight must be nonnegative')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    run_seed = seed if training_seed is None else int(training_seed)
    data_dir, model_dir = (Path(data_dir), Path(model_dir).resolve())
    suffix = '_NoIRM' if irm_base_weight == 0 else ''
    train_file = Path(train_file) if train_file is not None else data_dir / 'train_clean.npz'
    validation_file = Path(validation_file) if validation_file is not None else data_dir / 'validation_clean.npz'
    if train_file.resolve() == validation_file.resolve():
        raise ValueError('Training and validation must use different files')
    Training_Data = train_dataset if train_dataset is not None else AllDataset(train_file, mode='train')
    Validation_Data = validation_dataset if validation_dataset is not None else AllDataset(validation_file, mode='validation')
    if Training_Data.mode != 'train' or Validation_Data.mode != 'validation':
        raise ValueError('Training and calibration datasets have incorrect roles')
    check_disjoint_samples(Training_Data, Validation_Data)
    if len(Training_Data) < batch_size:
        raise ValueError('Training set must contain at least one full batch')
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir) if output_dir is not None else model_dir / 'evaluation'
    source_files = [Path(__file__), Path(__file__).with_name('direct_fusion_data.py'), Path(__file__).with_name('confidence_calibration.py'), Path(__file__).with_name('mdgr.py')]
    source_files += [Path(__file__).with_name(name) for name in ('z1_amp.py', 'z2_harm.py', 'z3_flicker.py', 'z4_tran.py')]
    if architecture_dir is not None:
        source_files += [Path(architecture_dir) / b / 'architecture.py' for b in BRANCH_NAMES]
    config = {'training_seed': run_seed, 'epochs': epochs, 'batch_size': batch_size, 'learning_rate': 0.001, 'irm_base_weight': irm_base_weight, 'device': str(device), 'threads': torch.get_num_threads(), 'cuda': device.type == 'cuda', 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU', 'selection': 'final_training_epoch', 'drop_last': True, 'architecture_dir': str(Path(architecture_dir).resolve()) if architecture_dir is not None else None, 'sources_sha256': {str(p.resolve()): _training_file_hash(p) for p in source_files}, 'train': _training_dataset_evidence(Training_Data), 'validation': _training_dataset_evidence(Validation_Data)}
    config_path = model_dir / 'training_config.json'
    if config_path.exists():
        if not resume:
            raise FileExistsError('Training run exists; use resume or a new model directory')
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Training config/data/source differs from saved run')
    else:
        if (model_dir / 'last.pt').exists() or (model_dir / 'training_record.json').exists():
            raise ValueError('Incomplete training configuration; preserve original run')
        _training_json(config_path, config)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(run_seed)
    nets = load_training_models(architecture_dir, device)
    initial = {b: _training_initial_hash(net) for b, net in nets.items()}
    names = {'amp': 'Causal_Amp', 'harm': 'Causal_harm', 'flicker': 'Causal_flicker', 'tran': 'Causal_tran'}
    calibration_path = model_dir / (calibration_filename or f'temperature_scaling{suffix}.json')
    if resume and (model_dir / 'training_record.json').exists():
        report = _verify_training_record(model_dir, config)
        for b, net in nets.items():
            net.load_state_dict(torch.load(report['model_files'][b], map_location=device, weights_only=True), strict=True)
            net.eval()
            net.zero_grad(set_to_none=True)
            net.requires_grad_(False)
        temperatures = load_temperatures(report['calibration_file'], nets)
    else:
        optimizers = {key: optim.Adam(net.parameters(), lr=0.001) for key, net in nets.items()}
        history, orders, start = ([], [], 0)
        if resume and (model_dir / 'last.pt').exists():
            saved = torch.load(model_dir / 'last.pt', map_location=device, weights_only=False)
            if saved['config'] != config or saved['initial_state_sha256'] != initial:
                raise ValueError('Checkpoint config/fresh initialization mismatch')
            start, history, orders = (saved['epoch'] + 1, saved['history'], saved['epoch_order_sha256'])
            expected_orders = [hashlib.sha256(np.random.default_rng(np.random.SeedSequence([run_seed, e, 20261029])).permutation(len(Training_Data)).astype('<i8').tobytes()).hexdigest() for e in range(start)]
            if not 1 <= start <= epochs or [r['epoch'] for r in history] != list(range(1, start + 1)) or orders != expected_orders:
                raise ValueError('Checkpoint history/sample order mismatch')
            for b in BRANCH_NAMES:
                nets[b].load_state_dict(saved['nets'][b], strict=True)
                optimizers[b].load_state_dict(saved['optimizers'][b])
            random.setstate(saved['rng']['python'])
            np.random.set_state(saved['rng']['numpy'])
            torch.set_rng_state(saved['rng']['torch'].cpu())
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all([v.cpu() for v in saved['rng']['cuda']])
        elif resume and any(model_dir.glob('Causal_*.pth')):
            raise ValueError('Training checkpoint missing; preserve incomplete run')
        criterion_task = nn.CrossEntropyLoss()
        training_started = time.monotonic()
        prior_wall = float(saved.get('training_wall_seconds', 0.0)) if resume and start else 0.0
        for epoch in range(start, epochs):
            started = time.monotonic()
            for net in nets.values():
                net.train()
            order = np.random.default_rng(np.random.SeedSequence([run_seed, epoch, 20261029])).permutation(len(Training_Data))
            orders.append(hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
            irm_weight = irm_base_weight * (2.0 / (1.0 + np.exp(-10.0 * epoch / epochs)) - 1.0)
            losses, batches = ({key: 0.0 for key in BRANCH_NAMES}, 0)
            for inputs, targets in _training_batches(Training_Data, order, batch_size, device):
                multiplicity = sum(((target != 0).int() for target in targets))
                for i, key in enumerate(BRANCH_NAMES):
                    optimizers[key].zero_grad(set_to_none=True)
                    logits = _training_logits(nets[key](inputs[i]), len(targets[i]), (4, 2, 2, 4)[i])
                    loss = compute_mdgr_loss(logits, targets[i], multiplicity, criterion_task, irm_weight)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    gradients = [p.grad.detach().reshape(-1) for p in nets[key].parameters() if p.grad is not None]
                    if not gradients or not torch.isfinite(torch.cat(gradients)).all():
                        raise ValueError('Missing/nonfinite training gradient')
                    optimizers[key].step()
                    losses[key] += float(loss.detach())
                batches += 1
            history.append({'epoch': epoch + 1, 'irm_weight': float(irm_weight), 'seconds': time.monotonic() - started, **{b: losses[b] / batches for b in BRANCH_NAMES}})
            checkpoint = {'config': config, 'epoch': epoch, 'history': history, 'epoch_order_sha256': orders, 'training_wall_seconds': prior_wall + time.monotonic() - training_started, 'initial_state_sha256': initial, 'nets': {b: net.state_dict() for b, net in nets.items()}, 'optimizers': {b: opt.state_dict() for b, opt in optimizers.items()}, 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
            _atomic_training_checkpoint(model_dir / 'last.pt', checkpoint)
            _training_json(model_dir / 'training_history.json', history)
            _training_json(model_dir / 'status.json', {'state': 'training', 'epoch_completed': epoch + 1, 'epochs': epochs, 'training_seed': run_seed, 'training_wall_seconds': prior_wall + time.monotonic() - training_started})
        _training_json(model_dir / 'training_history.json', history)
        model_files = {}
        for key, net in nets.items():
            net.eval()
            net.zero_grad(set_to_none=True)
            net.requires_grad_(False)
            path = model_dir / f'{names[key]}{suffix}.pth'
            torch.save(net.state_dict(), path)
            model_files[key] = str(path)
        _training_json(model_dir / 'pairing.json', {'initial_state_sha256': initial, 'epoch_order_sha256': orders})
        val_loader = DataLoader(Validation_Data, batch_size=batch_size, shuffle=False)
        temperatures = calibrate_branches(nets, val_loader, device, calibration_path)
        training_wall = prior_wall + time.monotonic() - training_started
        _training_json(model_dir / 'status.json', {'state': 'calibrated', 'epoch_completed': epochs, 'epochs': epochs, 'training_seed': run_seed, 'training_wall_seconds': training_wall})
        frozen = [model_dir / 'last.pt', config_path, model_dir / 'status.json', model_dir / 'training_history.json', model_dir / 'pairing.json', calibration_path]
        frozen += [Path(path) for path in model_files.values()]
        report = {'state': 'calibrated', 'training_seed': run_seed, 'epochs': epochs, 'irm_base_weight': irm_base_weight, 'config': config, 'model_variant': 'NoIRM' if suffix else 'IRM', 'model_files': model_files, 'calibration_file': str(calibration_path), 'temperatures': temperatures, 'test_evaluated': False, 'training_wall_seconds': training_wall, 'training_loop_seconds': sum((r['seconds'] for r in history)), 'frozen_files': {str(p): _training_file_hash(p) for p in frozen}}
        _training_json(model_dir / 'training_record.json', report)
    if not run_test:
        return report
    Testing_Data = make_test_dataset(data_dir)
    check_disjoint_samples(Training_Data, Validation_Data, Testing_Data)
    recorder = PredictionRecorder()
    branch_correct = {b: 0 for b in BRANCH_NAMES}
    with torch.inference_mode():
        for batch in DataLoader(Testing_Data, batch_size=batch_size, shuffle=False):
            logits = {b: _training_logits(nets[b](batch[2 * i].to(device)), len(batch[2 * i + 1]), (4, 2, 2, 4)[i]) for i, b in enumerate(BRANCH_NAMES)}
            preds, probabilities, confidence = calibrated_predictions(logits, temperatures)
            for i, b in enumerate(BRANCH_NAMES):
                branch_correct[b] += int((preds[b].cpu() == batch[2 * i + 1].cpu()).sum())
            recorder.add(batch[-1], map_preds_to_vector(*(preds[b] for b in BRANCH_NAMES)), probabilities, confidence)
    paths = recorder.save(output_dir / f'training_final{suffix}')
    with np.load(paths['npz'], allow_pickle=False) as saved:
        y_true, y_pred = (saved['y_true'], saved['y_pred'])
    metrics = {'exact_match': float(np.all(y_true == y_pred, axis=1).mean()), 'hamming_loss': float((y_true != y_pred).mean()), 'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)), 'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)), 'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'branch_accuracy': {b: branch_correct[b] / len(Testing_Data) for b in BRANCH_NAMES}}
    report = {**report, 'metrics': metrics, 'prediction_files': paths, 'test_evaluated': True, 'label_order': list(PAPER_LABEL_NAMES)}
    _training_json(output_dir / f'training_final{suffix}_metrics.json', report)
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=HERE.parent / 'generated_data')
    parser.add_argument('--output', type=Path, default=HERE.parent / 'outputs' / 'split_01_seed_42')
    parser.add_argument('--cache-root', type=Path, default=HERE.parent / 'outputs' / 'feature_cache')
    parser.add_argument('--architecture-dir', type=Path, default=HERE / 'architectures')
    parser.add_argument('--split', choices=('split_01', 'split_02', 'split_03'), default='split_01')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--mdgr-weight', type=float, default=0.5)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or args.seed < 0:
        parser.error('threads must be positive and seed nonnegative')
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        parser.error('CUDA is unavailable')
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    directory = args.data_root.resolve() / args.split
    train = prepare_cached_dataset(directory / 'train_clean.npz', 'train', args.cache_root, args.device)
    validation = prepare_cached_dataset(directory / 'validation_clean.npz', 'validation', args.cache_root, args.device)
    train_independent_decoupled(data_dir=directory, model_dir=args.output.resolve(), epochs=args.epochs, batch_size=args.batch_size, irm_base_weight=args.mdgr_weight, device=args.device, training_seed=args.seed, train_dataset=train, validation_dataset=validation, architecture_dir=args.architecture_dir.resolve(), run_test=False, resume=args.resume)
if __name__ == '__main__':
    main()
