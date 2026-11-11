import json
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent
SPECS = {'amp': {'public_class': 'AmpNet', 'shape': [2, 1200], 'classes': 4}, 'harm': {'public_class': 'HarmNet', 'shape': [604], 'classes': 2}, 'flicker': {'public_class': 'FlickNet', 'shape': [1, 1200], 'classes': 2}, 'tran': {'public_class': 'TranNet', 'shape': [2, 1200], 'classes': 4}}

def render_prompt(branch, mode, parent_architecture='', parent_metrics=None, pareto_context=None, memory_bank=None):
    spec = SPECS[branch]
    if mode.lower() not in ('exploration', 'exploitation'):
        raise ValueError('Search mode must be exploration or exploitation')
    text = (HERE / 'prompts' / f'{branch}.txt').read_text(encoding='utf-8')
    values = {'SEARCH_MODE': mode.upper(), 'PARENT_ARCHITECTURE': parent_architecture or 'None. Generate a new valid architecture.', 'PARENT_METRICS': json.dumps(parent_metrics, ensure_ascii=False, allow_nan=False), 'PARETO_CONTEXT': json.dumps(pareto_context or [], ensure_ascii=False, allow_nan=False), 'MEMORY_BANK': json.dumps(memory_bank or [], ensure_ascii=False, allow_nan=False)}
    text = re.sub('\\{\\{([A-Z_]+)\\}\\}', lambda match: values[match[1]], text)
    interface = f'\nExecution interface details:\n- Input tensor shape: [batch_size, {', '.join(map(str, spec['shape']))}].\n- Implement __init__(self, num_classes={spec['classes']}); num_classes is passed by keyword.\n- Return exactly (logits, features), both finite floating-point tensors; features must be two-dimensional and used by the classifier.\n- FLOPs use one sample: Conv/Linear/matmul multiply-add counts as 2; physical preprocessing, elementwise operations, normalization and pooling are excluded.\n- Use standard convolution, linear, pooling and elementwise building blocks. Do not use FFT, recurrent networks or Transformer/MultiheadAttention operators whose cost is not supported by this evaluator. Lightweight channel attention is allowed.\n'
    return text.replace('\nReturn only', interface + '\nReturn only', 1)

def seed_code(branch):
    spec = SPECS[branch]
    channels = 1 if branch == 'harm' else spec['shape'][0]
    prepare = 'spectrum = x[:, :600].unsqueeze(1)' if branch == 'harm' else 'spectrum = x'
    bypass = '\n        features = torch.cat([features, x[:, 600:]], dim=1)' if branch == 'harm' else ''
    features = 12 if branch == 'harm' else 8
    return f'import torch\nimport torch.nn as nn\n\nclass {spec['public_class']}(nn.Module):\n    def __init__(self, num_classes={spec['classes']}):\n        super().__init__()\n        self.encoder = nn.Sequential(nn.Conv1d({channels}, 8, 7, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1))\n        self.classifier = nn.Linear({features}, num_classes)\n\n    def forward(self, x):\n        {prepare}\n        features = self.encoder(spectrum).flatten(1){bypass}\n        logits = self.classifier(features)\n        return logits, features\n'
