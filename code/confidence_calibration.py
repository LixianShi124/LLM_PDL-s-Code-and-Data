import csv
import hashlib
import json
from numbers import Real
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch
BRANCH_NAMES = ('amp', 'harm', 'flicker', 'tran')
CLASSES = (4, 2, 2, 4)
PAPER_LABEL_NAMES = ('swell', 'sag', 'interruption', 'harmonic', 'flicker', 'impulsive', 'oscillatory', 'notch')
SCHEMA = 'pqd-temperature-calibration-v1'
BOUNDS = (0.01, 1000.0)
ROBUST_POLICY = {'method': 'group_cv_regularized_temperature_v2', 'temperature_bounds': [0.5, 10.0], 'regularization_grid': [0.001, 0.01, 0.1, 1.0], 'cv_folds': 5, 'cv_seed': 20261101, 'one_se_multiplier': 1.0, 'zero_error_identity': True}
HERE = Path(__file__).resolve().parent

def _temperature(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or (not np.isfinite(value)) or (value <= 0):
        raise ValueError('Temperature must be a finite positive scalar')
    return float(value)

def _keys(mapping, name):
    if not isinstance(mapping, dict) or set(mapping) != set(BRANCH_NAMES):
        raise ValueError(name + ' must contain exactly amp/harm/flicker/tran')

def _logits(value, classes=None):
    value = torch.as_tensor(value)
    if value.ndim != 2 or not len(value) or value.shape[1] < 2 or value.is_complex() or (not torch.isfinite(value).all()) or (classes is not None and value.shape[1] != classes):
        raise ValueError('Expected finite nonempty [N,C] logits with the correct class count')
    return value

def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

def _binary(value):
    value = np.asarray(_numpy(value))
    if value.ndim != 2 or value.shape[1] != 8 or (not len(value)) or (not np.isin(value, (0, 1)).all()):
        raise ValueError('Expected nonempty binary [N,8] paper labels')
    return value.astype(np.uint8, copy=True)

def to_paper_labels(y):
    return _binary(y)[:, [0, 1, 4, 2, 3, 5, 6, 7]]

def map_branch_predictions(p_amp, p_harm, p_flicker, p_tran):
    values = [torch.as_tensor(p) for p in (p_amp, p_harm, p_flicker, p_tran)]
    n, device = (len(values[0]) if values[0].ndim else 0, values[0].device)
    for p, count in zip(values, CLASSES):
        if p.ndim != 1 or not n or len(p) != n or (p.device != device) or p.is_complex() or (not torch.isfinite(p).all()) or (not ((p >= 0) & (p < count) & (p == p.long())).all()):
            raise ValueError('Invalid branch indices, lengths or devices')
    a, h, f, t = values
    return torch.stack((a == 1, a == 2, a == 3, h == 1, f == 1, t == 1, t == 2, t == 3), dim=1).long()

def fit_temperature(logits, targets):
    raw = _logits(logits).detach().cpu().double().numpy()
    y = np.asarray(_numpy(targets))
    if y.shape != (len(raw),) or not np.isfinite(y).all() or (not np.equal(y, np.floor(y)).all()) or np.any(y < 0) or np.any(y >= raw.shape[1]):
        raise ValueError('Invalid calibration class labels')
    y = y.astype(np.int64)

    def nll(temperature):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            scaled = raw / temperature if temperature >= 1 else raw
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            if temperature < 1:
                shifted = shifted / temperature
            losses = logsumexp(shifted, axis=1) - shifted[np.arange(len(y)), y]
            return float(np.sum(losses / len(y)))
    before = nll(1.0)
    if not np.isfinite(before):
        raise ValueError('Validation NLL cannot be represented as a finite float64 value')
    candidates = [1.0, *BOUNDS]
    if not np.all(raw == raw[:, :1]):
        result = minimize_scalar(lambda log_t: nll(float(np.exp(log_t))), bounds=np.log(BOUNDS), method='bounded', options={'xatol': 1e-09})
        if result.success and np.isfinite(result.x):
            candidates.append(float(np.clip(np.exp(result.x), *BOUNDS)))
    temperature = min(candidates, key=lambda t: (nll(t), abs(np.log(t)), t))
    return {'temperature': float(temperature), 'nll_before': before, 'nll_after': nll(temperature), 'n_samples': int(len(y))}

def fit_regularized_temperature(logits, targets, groups, policy=None):
    policy = json.loads(json.dumps(ROBUST_POLICY if policy is None else policy))
    if set(policy) != set(ROBUST_POLICY) or policy['method'] != ROBUST_POLICY['method']:
        raise ValueError('Unknown regularized temperature policy')
    low, high = policy['temperature_bounds']
    grid = policy['regularization_grid']
    if not 0 < low <= 1 <= high or low >= high or (not np.isfinite([low, high]).all()) or (not grid) or (not np.isfinite(grid).all()) or any((value < 0 for value in grid)) or (len(set(grid)) != len(grid)) or (type(policy['cv_folds']) is not int) or (policy['cv_folds'] < 2) or (type(policy['cv_seed']) is not int) or (not np.isfinite(policy['one_se_multiplier'])) or (policy['one_se_multiplier'] < 0):
        raise ValueError('Invalid regularized temperature bounds/grid/fold policy')
    original = _logits(logits)
    x = original.detach().cpu().double().numpy()
    y, groups = (np.asarray(_numpy(targets)), np.asarray(_numpy(groups)))
    if y.shape != (len(x),) or not np.isfinite(y).all() or (not np.equal(y, np.floor(y)).all()) or np.any(y < 0) or np.any(y >= x.shape[1]) or (groups.shape != (len(x),)):
        raise ValueError('Invalid targets or group/sample alignment')
    if groups.dtype.kind in 'biufc' and (not np.isfinite(groups).all()):
        raise ValueError('Nonfinite group identifiers')
    try:
        unique = np.unique(groups)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid group identifiers') from error
    y = y.astype(np.int64)

    def hard_losses(rows, labels, temperature):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            scaled = rows / temperature if temperature >= 1 else rows
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            if temperature < 1:
                shifted = shifted / temperature
            return logsumexp(shifted, axis=1) - shifted[np.arange(len(labels)), labels]
    baseline = hard_losses(x, y, 1.0)
    if not np.isfinite(baseline).all():
        raise ValueError('Nonfinite hard validation NLL')
    before = float(np.mean(baseline))
    errors = int(np.count_nonzero(original.argmax(1).detach().cpu().numpy() != y))
    result = {'temperature': 1.0, 'nll_before': before, 'nll_after': before, 'n_samples': len(x), 'n_errors': errors, 'n_groups': len(unique), 'method': policy['method'], 'temperature_bounds': [low, high], 'chosen_lambda': None, 'cv_assignment': [], 'candidates': [], 'selection_rule': 'paired sample-weighted OOF NLL gain > group-mean SE +1e-10; strongest lambda within one SE; full hard-NLL guard'}
    if errors == 0 and policy['zero_error_identity']:
        return {**result, 'reason': 'zero_validation_errors'}
    if len(unique) < 2:
        return {**result, 'reason': 'insufficient_groups'}
    shuffled = np.random.default_rng(policy['cv_seed']).permutation(unique)
    folds = np.array_split(shuffled, min(policy['cv_folds'], len(unique)))
    masks = [np.isin(groups, fold) for fold in folds]
    result['cv_assignment'] = [{'fold': i, 'validation_groups': fold.tolist(), 'training_group_count': len(unique) - len(fold)} for i, fold in enumerate(folds)]

    def fit(rows, labels, strength):

        def objective(t):
            return float(np.mean(hard_losses(rows, labels, t))) + strength * np.log(t) ** 2
        opt = minimize_scalar(lambda log_t: objective(float(np.exp(log_t))), bounds=np.log([low, high]), method='bounded', options={'xatol': 1e-09})
        candidates = [1.0, low, high]
        if opt.success and np.isfinite(opt.x):
            candidates.append(float(np.clip(np.exp(opt.x), low, high)))
        return min(candidates, key=lambda t: (objective(t), abs(np.log(t)), t))
    for strength in grid:
        oof = np.empty(len(x))
        temperatures = []
        coverage = np.zeros(len(x), dtype=np.int8)
        for mask in masks:
            t = fit(x[~mask], y[~mask], strength)
            temperatures.append(float(t))
            oof[mask] = hard_losses(x[mask], y[mask], t)
            coverage[mask] += 1
        if not np.all(coverage == 1):
            raise ValueError('Incomplete or repeated OOF assignment')
        gain = baseline - oof
        group_gains = np.array([gain[groups == group].mean() for group in unique])
        se = float(group_gains.std(ddof=1) / np.sqrt(len(unique)))
        result['candidates'].append({'lambda': strength, 'oof_nll': float(oof.mean()), 'paired_mean_gain': float(gain.mean()), 'paired_group_gain_se': se, 'group_mean_gains': [{'group': group, 'gain': float(value), 'n_samples': int((groups == group).sum())} for group, value in zip(unique.tolist(), group_gains)], 'oof_sample_count': int(coverage.sum()), 'fold_temperatures': temperatures})
    best = max(result['candidates'], key=lambda row: (row['paired_mean_gain'], row['lambda']))
    allowance = policy['one_se_multiplier'] * best['paired_group_gain_se']
    result.update(best_oof_gain=best['paired_mean_gain'], best_oof_group_se=best['paired_group_gain_se'])
    if best['paired_mean_gain'] <= allowance + 1e-10:
        return {**result, 'reason': 'identity_no_grouped_evidence'}
    chosen = max((row for row in result['candidates'] if row['paired_mean_gain'] >= best['paired_mean_gain'] - allowance), key=lambda row: row['lambda'])
    temperature = float(fit(x, y, chosen['lambda']))
    after = float(hard_losses(x, y, temperature).mean())
    if not np.isfinite(after) or after > before:
        return {**result, 'reason': 'full_validation_nll_guard', 'attempted_lambda': chosen['lambda']}
    return {**result, 'temperature': temperature, 'nll_after': after, 'chosen_lambda': chosen['lambda'], 'reason': 'regularized_group_cv_evidence', 'full_refit_temperature': temperature}

def calibrated_predictions(logits_by_branch, temperatures):
    _keys(logits_by_branch, 'Logits')
    _keys(temperatures, 'Temperatures')
    predictions, probabilities, confidences = ({}, {}, [])
    n, device = (None, None)
    for branch, classes in zip(BRANCH_NAMES, CLASSES):
        raw = _logits(logits_by_branch[branch], classes)
        if n is not None and (len(raw) != n or raw.device != device):
            raise ValueError('All branches must have identical sample count and device')
        n, device = (len(raw), raw.device)
        temperature = _temperature(temperatures[branch])
        predictions[branch] = raw.argmax(dim=1)
        scaled = raw.double() / temperature if temperature >= 1 else raw.double()
        shifted = scaled - scaled.max(dim=1, keepdim=True).values
        if temperature < 1:
            shifted = shifted / temperature
        probabilities[branch] = torch.softmax(shifted, dim=1)
        if not torch.isfinite(probabilities[branch]).all():
            raise ValueError('Nonfinite calibrated probability')
        confidences.append(probabilities[branch].gather(1, predictions[branch].unsqueeze(1)).squeeze(1))
    return (predictions, probabilities, torch.stack(confidences, dim=1))

def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _preprocessing_hashes():
    return {name: _file_hash(HERE / name) for name in ('z1_amp.py', 'z2_harm.py', 'z3_flicker.py', 'z4_tran.py')}

def _model_hashes(nets):
    _keys(nets, 'Models')
    result = {}
    for branch in BRANCH_NAMES:
        digest = hashlib.sha256()
        for name, value in sorted(nets[branch].state_dict().items()):
            value = value.detach().cpu().contiguous()
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite model state')
            digest.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[branch] = digest.hexdigest()
    return result

def _metadata():
    return {'schema': SCHEMA, 'branch_order': list(BRANCH_NAMES), 'class_counts': list(CLASSES), 'paper_label_order': list(PAPER_LABEL_NAMES), 'branch_class_names': {'amp': ['none', 'swell', 'sag', 'interruption'], 'harm': ['none', 'harmonic'], 'flicker': ['none', 'flicker'], 'tran': ['none', 'impulsive', 'oscillatory', 'notch']}}

def _validation_cohort(dataset, fitted_count):
    paths = getattr(dataset, 'file_paths', [])
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(path).resolve() for path in paths]
    result = {'dataset_mode': 'validation', 'dataset_n_samples': len(dataset), 'fitted_n_samples': fitted_count, 'file_paths': [str(path) for path in paths], 'file_sha256': {str(path): _file_hash(path) for path in paths if path.is_file()}}
    ids = getattr(dataset, 'sample_ids', None)
    if ids is not None:
        ids = np.asarray(_numpy(ids))
        encoded = json.dumps(ids.tolist(), ensure_ascii=False, separators=(',', ':'), default=str).encode('utf-8')
        result.update(sample_id_sha256=hashlib.sha256(encoded).hexdigest(), sample_id_count=int(ids.size), sample_id_scope='validation dataset order; fitted loader count recorded separately')
    return result

def _save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def calibrate_branches(nets, val_loader, device, output_path, method='nll'):
    if method not in ('nll', 'regularized', 'legacy'):
        raise ValueError('Unknown calibration method')
    if getattr(val_loader.dataset, 'mode', None) != 'validation':
        raise ValueError("Temperature fitting requires dataset.mode == 'validation'")
    before = _model_hashes(nets)
    values, labels = ({b: [] for b in BRANCH_NAMES}, {b: [] for b in BRANCH_NAMES})
    for model in nets.values():
        model.eval()
    with torch.inference_mode():
        for batch in val_loader:
            if len(batch) != 9:
                raise ValueError('Expected four x/y pairs plus multilabel targets')
            for index, (branch, classes) in enumerate(zip(BRANCH_NAMES, CLASSES)):
                output = nets[branch](batch[2 * index].to(device))
                output = output[0] if isinstance(output, (tuple, list)) else output
                values[branch].append(_logits(output, classes).detach().cpu().double())
                labels[branch].append(torch.as_tensor(batch[2 * index + 1]).detach().cpu())
    if not values['amp']:
        raise ValueError('Empty validation cohort')
    if _model_hashes(nets) != before:
        raise ValueError('Model state changed during calibration inference')
    targets = {b: torch.cat(labels[b]) for b in BRANCH_NAMES}
    groups = targets['amp'] * 16 + targets['harm'] * 8 + targets['flicker'] * 4 + targets['tran']
    fits = {b: fit_regularized_temperature(torch.cat(values[b]), targets[b], groups) if method == 'regularized' else fit_temperature(torch.cat(values[b]), targets[b]) for b in BRANCH_NAMES}
    if len({fit['n_samples'] for fit in fits.values()}) != 1:
        raise ValueError('Different branch validation sample counts')
    temperatures = {b: fits[b]['temperature'] for b in BRANCH_NAMES}
    _save_json(output_path, {**_metadata(), 'role': 'validation', 'method': ROBUST_POLICY['method'] if method == 'regularized' else 'temperature_scaling', 'policy': ROBUST_POLICY if method == 'regularized' else None, 'calibration_source_sha256': _file_hash(__file__), 'temperature_bounds': ROBUST_POLICY['temperature_bounds'] if method == 'regularized' else list(BOUNDS), 'temperatures': temperatures, 'fits': fits, 'model_state_sha256': before, 'preprocessing_sha256': _preprocessing_hashes(), 'validation_cohort': _validation_cohort(val_loader.dataset, fits['amp']['n_samples']), 'decision_rule': 'branch_argmax', 'nll_partition': 'validation'})
    return temperatures

def load_temperatures(path, nets):
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    if any((saved.get(key) != value for key, value in _metadata().items())) or saved.get('role') != 'validation':
        raise ValueError('Calibration schema/order/class mismatch')
    if saved.get('preprocessing_sha256') != _preprocessing_hashes():
        raise ValueError('Calibration preprocessing source mismatch')
    if saved.get('model_state_sha256') != _model_hashes(nets):
        raise ValueError('Calibration model state mismatch')
    temperatures = saved.get('temperatures')
    _keys(temperatures, 'Saved temperatures')
    return {branch: _temperature(temperatures[branch]) for branch in BRANCH_NAMES}

class PredictionRecorder:

    def __init__(self):
        self.batches = []

    def add(self, labels, predictions, probabilities, confidences):
        truth = _binary(labels)
        if isinstance(predictions, dict):
            _keys(predictions, 'Predictions')
            predictions = map_branch_predictions(*(predictions[b] for b in BRANCH_NAMES))
        pred = _binary(predictions)
        q = torch.as_tensor(confidences).detach().cpu().double().clone()
        if pred.shape != truth.shape or q.shape != (len(truth), 4) or (not torch.isfinite(q).all()) or (not ((q >= 0) & (q <= 1)).all()):
            raise ValueError('Invalid aligned labels, predictions or q')
        _keys(probabilities, 'Probabilities')
        probs = {}
        for index, (branch, count) in enumerate(zip(BRANCH_NAMES, CLASSES)):
            p = torch.as_tensor(probabilities[branch]).detach().cpu().double().clone()
            if p.shape != (len(truth), count) or not torch.isfinite(p).all() or (not ((p >= 0) & (p <= 1)).all()) or (not torch.allclose(p.sum(1), torch.ones(len(p), dtype=torch.float64), atol=1e-08, rtol=1e-06)) or (not torch.allclose(q[:, index], p.max(1).values, atol=1e-08, rtol=1e-06)):
                raise ValueError('Invalid normalized class probabilities or branch confidence')
            probs[branch] = p
        self.batches.append((truth, pred, probs, q))

    def save(self, output_prefix):
        if not self.batches:
            raise ValueError('No prediction records to save')
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        paths = {key: str(Path(str(prefix) + suffix).resolve()) for key, suffix in (('npz', '.npz'), ('csv', '.csv'), ('metadata', '.metadata.json'))}
        y_true = np.concatenate([batch[0] for batch in self.batches])
        y_pred = np.concatenate([batch[1] for batch in self.batches])
        q = torch.cat([batch[3] for batch in self.batches]).numpy()
        probabilities = {b: torch.cat([batch[2][b] for batch in self.batches]).numpy() for b in BRANCH_NAMES}
        np.savez_compressed(paths['npz'], y_true=y_true, y_pred=y_pred, q=q, **{b + '_probabilities': p for b, p in probabilities.items()})
        fields = [prefix + label for label in PAPER_LABEL_NAMES for prefix in ('true_', 'pred_')]
        fields += [b + '_q' for b in BRANCH_NAMES]
        fields += [f'{b}_p{c}' for b, count in zip(BRANCH_NAMES, CLASSES) for c in range(count)]
        with Path(paths['csv']).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for i in range(len(y_true)):
                row = [v for pair in zip(y_true[i], y_pred[i]) for v in pair] + q[i].tolist()
                writer.writerow(row + [v for b in BRANCH_NAMES for v in probabilities[b][i]])
        _save_json(paths['metadata'], {**_metadata(), 'n_samples': len(y_true), 'confidence_order': list(BRANCH_NAMES), 'confidence_definition': 'one selected-class probability per branch; not joint probability', 'files_sha256': {key: _file_hash(paths[key]) for key in ('npz', 'csv')}})
        return paths
