import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.flop_counter import FlopCounterMode
from pgenas.candidate_guard import validate_candidate_static, branch_spec
FLOP_CONVENTION = 'One sample; Conv/Linear/matmul multiply-add=2 FLOPs; bias, elementwise, normalization, pooling and physical preprocessing excluded.'

def validate_operator_space(code):
    tree = ast.parse(code)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split('.')[0]] = name.name if name.asname else name.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                aliases[name.asname or name.name] = node.module + '.' + name.name

    def canonical(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = canonical(node.value)
            return base + '.' + node.attr if base else None
        return None
    layers = set('Module Sequential ModuleList ModuleDict Parameter ParameterList ParameterDict\n        Conv1d Linear BatchNorm1d LayerNorm GroupNorm InstanceNorm1d\n        Dropout Dropout1d Identity Flatten Unflatten ReLU ReLU6 LeakyReLU PReLU ELU GELU SiLU\n        Sigmoid Tanh Hardsigmoid Hardswish Softmax LogSoftmax\n        MaxPool1d AvgPool1d AdaptiveAvgPool1d AdaptiveMaxPool1d\n        ConstantPad1d ReflectionPad1d ReplicationPad1d'.split())
    functional = set('conv1d linear relu relu6 leaky_relu prelu elu gelu silu sigmoid tanh\n        hardsigmoid hardswish softmax log_softmax dropout dropout1d batch_norm layer_norm\n        group_norm instance_norm normalize pad interpolate max_pool1d avg_pool1d\n        adaptive_avg_pool1d adaptive_max_pool1d'.split())
    functions = set('cat concat concatenate stack flatten reshape transpose permute squeeze unsqueeze\n        split chunk unbind mean sum max min amax amin std var abs sqrt square pow exp log log1p\n        tanh sigmoid relu softmax log_softmax clamp where zeros ones empty full zeros_like ones_like\n        empty_like full_like arange linspace tensor as_tensor rand randn randint finfo iinfo\n        isfinite isnan isinf no_grad enable_grad matmul mm bmm einsum Tensor Size'.split())

    def check(path, called=False):
        if not path:
            return
        if path.startswith(('torch.fft', 'torch.linalg')):
            raise ValueError('Unsupported spectral/linalg operator: ' + path)
        prefix = 'torch.nn.functional.'
        if path.startswith(prefix) and path[len(prefix):] not in functional:
            raise ValueError('Unsupported functional operator: ' + path)
        if called and path.startswith('torch.nn.') and (not path.startswith((prefix, 'torch.nn.init.'))):
            if path[len('torch.nn.'):] not in layers:
                raise ValueError('Unsupported network layer: ' + path)
        if called and path.startswith('torch.') and (path.count('.') == 1) and (path[6:] not in functions):
            raise ValueError('Unsupported tensor function: ' + path)
    for path in aliases.values():
        check(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            check(canonical(node))
        elif isinstance(node, ast.Call):
            check(canonical(node.func), called=True)

def profile_model(model, input_shape):
    unsupported = (nn.RNNBase, nn.RNNCellBase, nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder, nn.Embedding, nn.EmbeddingBag, nn.Bilinear)
    if any((isinstance(m, unsupported) for m in model.modules())):
        raise ValueError('Unsupported expensive module outside the lightweight Conv/MLP search space')
    params = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    if trainable != params or params == 0:
        raise ValueError('All candidate parameters must be trainable')
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        output = model(torch.randn(1, *input_shape))
    flops = int(counter.get_total_flops())
    if flops <= 0:
        raise ValueError('No supported Conv/Linear computation was counted')
    return {'params': params, 'flops': flops, 'flop_convention': FLOP_CONVENTION, 'input_shape': [1, *input_shape], 'counted_ops': sorted(map(str, counter.flop_counts['Global']))}

def signature(model):
    return [(n, tuple(p.shape), id(p)) for n, p in model.named_parameters()]

def logits(model, x, classes, expected):
    output = model(x)
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError('Network must return (logits, features)')
    output, features = output
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] != len(x) or (features.shape[1] < 1) or (not features.is_floating_point()) or (not bool(torch.isfinite(features).all())):
        raise ValueError('Invalid learned feature tensor')
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise ValueError('Logits must be a floating-point tensor')
    if signature(model) != expected:
        raise ValueError('Parameters changed during forward')
    if output.shape != (len(x), classes) or not bool(torch.isfinite(output).all()):
        raise ValueError('Wrong output shape or nonfinite logits')
    return output

def validation(model, loader, device, classes, expected):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    predictions = []
    with torch.no_grad():
        for x, y in loader:
            x, y = (x.to(device), y.to(device))
            out = logits(model, x, classes, expected)
            predicted = out.argmax(1)
            correct += int((predicted == y).sum())
            total += len(y)
            loss_sum += float(nn.functional.cross_entropy(out, y, reduction='sum'))
            predictions.append(predicted.cpu().numpy())
    return ({'accuracy': correct / total, 'cross_entropy': loss_sum / total, 'count': total}, np.concatenate(predictions))

def run(job):
    torch.set_num_threads(job['threads'])
    seed = job['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    code_path = Path(job['code_path'])
    code = code_path.read_text(encoding='utf-8')
    checked = validate_candidate_static(code, job['branch'])
    if not checked['accepted']:
        raise ValueError('Invalid candidate: ' + str(checked['errors']))
    validate_operator_space(code)
    spec = branch_spec(job['branch'])
    namespace = {'__name__': 'pareto_candidate'}
    exec(compile(code, str(code_path), 'exec'), namespace)
    model = namespace[spec['class_name']](num_classes=spec['num_classes'])
    expected = signature(model)
    params = sum((p.numel() for p in model.parameters()))
    base = {'training_started': False, 'test_data_accessed': False, 'params': params}
    if params > job['max_params']:
        return {**base, 'status': 'resource_rejected', 'error': 'Parameter budget exceeded'}
    measured = profile_model(model, spec['input_shape'])
    if signature(model) != expected:
        raise ValueError('Parameters changed during profiling')
    base.update(measured)
    if measured['flops'] > job['max_flops']:
        return {**base, 'status': 'resource_rejected', 'error': 'FLOPs budget exceeded'}
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    probe = logits(model, torch.randn(4, *spec['input_shape']), spec['num_classes'], expected)
    nn.functional.cross_entropy(probe, torch.arange(4) % spec['num_classes']).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all((bool(torch.isfinite(g).all()) for g in grads)):
        raise ValueError('Missing or nonfinite gradients')
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    device = torch.device(job['device'])
    model.to(device)
    expected = signature(model)
    datasets = {}
    for role in ('train', 'validation'):
        x = np.load(job[role + '_x'], mmap_mode='r', allow_pickle=False)
        y = np.load(job[role + '_y'], allow_pickle=False)
        if len(x) != len(y) or not np.isfinite(x).all() or len(x) < 2:
            raise ValueError('Invalid training/validation arrays')
        datasets[role] = TensorDataset(torch.from_numpy(np.array(x, copy=True)), torch.from_numpy(y.astype('int64')))
    drop_last = len(datasets['train']) % job['batch_size'] == 1
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(datasets['train'], batch_size=job['batch_size'], shuffle=True, generator=generator, drop_last=drop_last)
    valid = DataLoader(datasets['validation'], batch_size=job['batch_size'], shuffle=False)
    if not len(train):
        raise ValueError('No training batches')
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=job['learning_rate'])
    history = []
    best = None
    best_state = None
    best_predictions = None
    for epoch in range(1, job['epochs'] + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, y in train:
            x, y = (x.to(device), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            out = logits(model, x, spec['num_classes'], expected)
            loss = nn.functional.cross_entropy(out, y)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            if any((p.grad is not None and (not bool(torch.isfinite(p.grad).all())) for p in model.parameters())):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)
        metrics, predictions = validation(model, valid, device, spec['num_classes'], expected)
        history.append({'epoch': epoch, 'train_ce': loss_sum / count, **metrics})
        if best is None or metrics['accuracy'] > best['accuracy']:
            best = {'epoch': epoch, **metrics}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_predictions = predictions
    output = Path(job['output'])
    torch.save({'state_dict': best_state, 'branch': job['branch'], 'class_name': spec['class_name'], 'best_validation': best, 'transient_preprocessing': job['transient_preprocessing'], 'architecture_sha256': hashlib.sha256(code_path.read_bytes()).hexdigest(), 'job': job}, output / 'best_validation.pt')
    np.savez_compressed(output / 'validation_predictions.npz', y_true=np.load(job['validation_y']), y_pred=best_predictions)
    return {**base, 'status': 'trained', 'training_started': True, 'accuracy': best['accuracy'], 'best_validation': best, 'history': history, 'training_count': len(datasets['train']), 'validation_count': len(datasets['validation']), 'dropped_last_training_batch': drop_last}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    args = p.parse_args()
    job = json.loads(args.job.read_text(encoding='utf-8'))
    start = time.monotonic()
    try:
        result = run(job)
    except Exception as e:
        result = {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc(limit=6), 'test_data_accessed': False}
    result['elapsed_seconds'] = time.monotonic() - start
    Path(job['output'], 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if result['status'] == 'failed':
        raise SystemExit(1)
if __name__ == '__main__':
    main()
