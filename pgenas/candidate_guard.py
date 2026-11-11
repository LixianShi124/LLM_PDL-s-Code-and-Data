from __future__ import annotations
import ast
import hashlib
import re
from typing import Any
from pgenas.paper_prompts import SPECS
BRANCHES = {b: dict(class_name=s['public_class'], input_shape=tuple(s['shape']), num_classes=s['classes']) for b, s in SPECS.items()}
ALIASES = {'amplitude': 'amp', 'harmonic': 'harm', 'flick': 'flicker', 'transient': 'tran'}
MAX_CODE_BYTES = 100000
MAX_AST_NODES = 20000
SAFE_BUILTINS = {'super', 'len', 'range', 'enumerate', 'zip', 'sum', 'min', 'max', 'abs', 'int', 'float', 'bool', 'str', 'round', 'tuple', 'list', 'dict', 'set', 'isinstance', 'all', 'any', 'sorted', 'reversed', 'iter', 'next', 'Exception', 'ValueError', 'TypeError', 'RuntimeError', 'AssertionError', 'NotImplementedError'}
NN_CLASSES = set('Module Sequential ModuleList ModuleDict Parameter ParameterList ParameterDict\n    Conv1d Conv2d Conv3d ConvTranspose1d ConvTranspose2d ConvTranspose3d Linear Bilinear\n    Embedding EmbeddingBag BatchNorm1d BatchNorm2d BatchNorm3d LayerNorm GroupNorm\n    InstanceNorm1d InstanceNorm2d InstanceNorm3d RMSNorm LocalResponseNorm\n    Dropout Dropout1d Dropout2d Dropout3d AlphaDropout FeatureAlphaDropout\n    ReLU ReLU6 LeakyReLU PReLU RReLU ELU CELU SELU GELU SiLU Hardswish Hardsigmoid\n    Hardtanh Sigmoid Tanh Softmax LogSoftmax Softmin GLU Mish Softplus Softshrink\n    Hardshrink Tanhshrink Threshold Identity Flatten Unflatten\n    MaxPool1d MaxPool2d MaxPool3d AvgPool1d AvgPool2d AvgPool3d\n    AdaptiveAvgPool1d AdaptiveAvgPool2d AdaptiveAvgPool3d\n    AdaptiveMaxPool1d AdaptiveMaxPool2d AdaptiveMaxPool3d\n    Upsample UpsamplingNearest2d UpsamplingBilinear2d\n    ConstantPad1d ConstantPad2d ConstantPad3d ReflectionPad1d ReflectionPad2d\n    ReplicationPad1d ReplicationPad2d ReplicationPad3d CircularPad1d CircularPad2d\n    RNN GRU LSTM RNNCell GRUCell LSTMCell MultiheadAttention\n    Transformer TransformerEncoder TransformerDecoder TransformerEncoderLayer TransformerDecoderLayer\n    PixelShuffle PixelUnshuffle ChannelShuffle'.split())
TORCH_FUNCTIONS = set('cat concat concatenate stack hstack vstack dstack split chunk unbind\n    flatten reshape transpose permute squeeze unsqueeze mean sum max min amax amin std var\n    abs absolute sqrt square pow exp log log1p log2 log10 sin cos tanh sigmoid relu softmax\n    log_softmax clamp clamp_min clamp_max clip where zeros ones empty full zeros_like\n    ones_like empty_like full_like arange linspace tensor as_tensor eye matmul bmm mm\n    einsum outer inner dot diag diagonal triu tril tril_indices triu_indices roll flip\n    repeat_interleave finfo iinfo isfinite isnan isinf no_grad enable_grad\n    rand randn randint rand_like randn_like bernoulli Tensor Size'.split())
TORCH_CONSTANTS = set('float float16 float32 float64 double half bfloat16 int int8 int16 int32 int64\n    long short uint8 bool complex64 complex128 pi inf nan'.split())
FUNCTIONAL = set('conv1d conv2d conv3d conv_transpose1d conv_transpose2d conv_transpose3d linear bilinear\n    relu relu6 leaky_relu prelu rrelu elu celu selu gelu silu hardswish hardsigmoid\n    hardtanh sigmoid tanh softmax log_softmax softmin glu mish softplus softshrink hardshrink tanhshrink\n    max_pool1d max_pool2d max_pool3d avg_pool1d avg_pool2d avg_pool3d\n    adaptive_avg_pool1d adaptive_avg_pool2d adaptive_avg_pool3d\n    adaptive_max_pool1d adaptive_max_pool2d adaptive_max_pool3d\n    dropout dropout1d dropout2d dropout3d alpha_dropout feature_alpha_dropout\n    batch_norm layer_norm group_norm instance_norm normalize local_response_norm\n    interpolate pad unfold fold embedding embedding_bag cosine_similarity pairwise_distance\n    scaled_dot_product_attention'.split())
INITIALIZERS = set('uniform_ normal_ constant_ ones_ zeros_ eye_ dirac_ xavier_uniform_ xavier_normal_\n    kaiming_uniform_ kaiming_normal_ orthogonal_ sparse_ trunc_normal_ calculate_gain'.split())
FFT_FUNCTIONS = set('fft ifft rfft irfft fftn ifftn rfftn irfftn fftfreq rfftfreq fftshift ifftshift'.split())
LINALG_FUNCTIONS = {'norm', 'vector_norm', 'matrix_norm'}
MATH_FUNCTIONS = set('sqrt ceil floor log log1p log2 log10 exp expm1 pow sin cos tan tanh sinh cosh\n    asin acos atan atan2 asinh acosh atanh fabs copysign hypot isfinite isnan isinf prod fsum\n    pi e tau inf nan'.split())
TENSOR_METHODS = set('size dim ndimension numel view reshape flatten transpose permute contiguous\n    unsqueeze squeeze expand expand_as repeat repeat_interleave unfold mean max min sum amax amin\n    std var abs sqrt pow square exp log log1p log2 log10 sin cos tanh sigmoid relu softmax log_softmax\n    clamp clamp_min clamp_max float double half bfloat16 to type_as clone detach chunk split unbind\n    narrow select index_select gather scatter scatter_add masked_fill where flip roll sort argsort\n    topk cumsum cumprod triu tril diagonal norm item add mul div sub add_ mul_ div_ sub_ relu_\n    zero_ fill_ normal_ uniform_ copy_ new_zeros new_ones new_empty new_full\n    parameters named_parameters modules named_modules children register_buffer register_parameter\n    append extend insert keys items get values'.split())
TENSOR_FIELDS = set('shape ndim device dtype T H real imag values indices weight bias data\n    running_mean running_var num_batches_tracked training in_channels out_channels kernel_size\n    stride padding dilation groups num_features num_classes batch_first hidden_size input_size\n    num_layers bidirectional eps p normalized_shape requires_grad'.split())
MODULE_PREFIXES = {'torch', 'torch.nn', 'torch.nn.functional', 'torch.nn.init', 'torch.fft', 'torch.linalg', 'math'}
ALLOWED_IMPORTS = {'torch', 'torch.nn', 'torch.nn.functional', 'torch.nn.init', 'torch.fft', 'torch.linalg', 'math'}

def branch_spec(branch: str) -> dict:
    branch = ALIASES.get(branch, branch)
    if branch not in BRANCHES:
        raise ValueError(f'Unknown branch: {branch}')
    return {'branch': branch, **BRANCHES[branch]}

def extract_python_code(text: str) -> str:
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_CODE_BYTES:
        raise ValueError('Candidate must be a text string within the code-size limit')
    fences = re.findall('```(?:python|py)?\\s*\\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if len(fences) > 1:
        raise ValueError('Multiple code blocks are ambiguous')
    code = (fences[0] if fences else text).strip() + '\n'
    if not code.strip():
        raise ValueError('Empty candidate')
    return code

class _DropDocstrings(ast.NodeTransformer):

    def _strip(self, node: Any) -> Any:
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node
    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip

def normalized_code_hash(code: str) -> str:
    tree = _DropDocstrings().visit(ast.parse(code))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode('utf-8')).hexdigest()

class _StaticGuard(ast.NodeVisitor):

    def __init__(self, tree: ast.Module, spec: dict):
        self.tree, self.spec = (tree, spec)
        self.errors = []
        self.aliases = {}
        self.classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        self.methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.bound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param))}
        self.bound.update((node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)))
        self.bound.update((node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))))
        self.self_fields = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and (node.value.id == 'self')}

    def error(self, node: ast.AST, message: str) -> None:
        self.errors.append(f'L{getattr(node, 'lineno', 0)}: {message}')

    def path(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return self.aliases[node.id]
        if isinstance(node, ast.Attribute):
            parent = self.path(node.value)
            return parent + '.' + node.attr if parent else None
        return None

    def allowed_path(self, path: str) -> bool:
        if path in MODULE_PREFIXES:
            return True
        sections = path.rsplit('.', 1)
        if len(sections) != 2:
            return False
        prefix, name = sections
        choices = {'torch': TORCH_FUNCTIONS | TORCH_CONSTANTS, 'torch.nn': NN_CLASSES, 'torch.nn.functional': FUNCTIONAL, 'torch.nn.init': INITIALIZERS, 'torch.fft': FFT_FUNCTIONS, 'torch.linalg': LINALG_FUNCTIONS, 'math': MATH_FUNCTIONS}
        return name in choices.get(prefix, set())

    def _import_alias(self, node: ast.AST, alias: str, path: str) -> None:
        if alias.startswith('_') or alias in SAFE_BUILTINS:
            self.error(node, 'Unsafe import alias')
        self.aliases[alias] = path
        self.bound.add(alias)

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            if item.name not in ALLOWED_IMPORTS:
                self.error(node, f'Import is not allowed: {item.name}')
            self._import_alias(node, item.asname or item.name.split('.')[0], item.name if item.asname else item.name.split('.')[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module not in ALLOWED_IMPORTS:
            self.error(node, 'Only explicit torch/math imports are allowed')
        for item in node.names:
            path = f'{node.module}.{item.name}'
            if item.name == '*' or not self.allowed_path(path):
                self.error(node, f'Imported member is not allowed: {path}')
            self._import_alias(node, item.asname or item.name, path)

    def check(self) -> list[str]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                self.visit_Import(node)
            elif isinstance(node, ast.ImportFrom):
                self.visit_ImportFrom(node)
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError, RecursionError):
                    self.error(node, 'Top-level assignments must be literal constants')
                continue
            self.error(node, 'Only architecture definitions/imports/literal constants are permitted at module level')
        main = self.classes.get(self.spec['class_name'])
        if main is None:
            self.error(self.tree, f'Missing required class {self.spec['class_name']}')
        else:
            constructors = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == '__init__']
            forwards = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == 'forward']
            if len(constructors) != 1 or len(forwards) != 1:
                self.error(main, 'Main class must define one __init__ and one forward')
            else:
                constructor = constructors[0]
                arguments = constructor.args
                names = [arg.arg for arg in arguments.args]
                required_count = len(arguments.args) - len(arguments.defaults)
                if not names or names[0] != 'self' or 'num_classes' not in names or (required_count > 2) or arguments.vararg or arguments.kwarg or arguments.posonlyargs or arguments.kwonlyargs:
                    self.error(constructor, 'Constructor must accept num_classes by keyword and no other required arguments')
                required_names = names[:required_count]
                if any((name not in ('self', 'num_classes') for name in required_names)):
                    self.error(constructor, 'Unknown required constructor argument')
                forward = forwards[0].args
                if len(forward.args) - len(forward.defaults) > 2 or len(forward.args) < 2 or forward.vararg or forward.kwarg or forward.posonlyargs or forward.kwonlyargs:
                    self.error(forwards[0], 'forward must be callable with one input tensor')
        self.visit(self.tree)
        return list(dict.fromkeys(self.errors))

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal, ast.Delete, ast.While)):
            self.error(node, f'Unsupported architecture construct: {type(node).__name__}')
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith('__'):
            self.error(node, 'Dunder names/introspection are not allowed')
        if isinstance(node.ctx, ast.Store) and (node.id in self.aliases or node.id in SAFE_BUILTINS):
            self.error(node, 'Imports and safe builtins cannot be rebound')
        if isinstance(node.ctx, ast.Load) and node.id not in self.bound and (node.id not in SAFE_BUILTINS):
            self.error(node, f'Unbound/unsafe name: {node.id}')

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg.startswith('__') or node.arg in self.aliases or node.arg in SAFE_BUILTINS:
            self.error(node, 'Unsafe argument name')
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.decorator_list or node.keywords or len(node.bases) != 1:
            self.error(node, 'Classes must directly inherit nn.Module without decorators/metaclass')
        elif self.path(node.bases[0]) != 'torch.nn.Module' and (not (isinstance(node.bases[0], ast.Name) and node.bases[0].id in self.classes)):
            self.error(node, 'Only nn.Module/local module helper inheritance is allowed')
        if node.name.startswith('_') or node.name in self.aliases or node.name in SAFE_BUILTINS:
            self.error(node, 'Unsafe class name')
        other_model_names = {item['class_name'] for item in BRANCHES.values()} - {self.spec['class_name']}
        if node.name in other_model_names:
            self.error(node, "Candidate defines another branch's main class")
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef):
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                try:
                    ast.literal_eval(statement.value)
                except (ValueError, TypeError, RecursionError):
                    self.error(statement, 'Class-body values must be literal constants')
            else:
                self.error(statement, 'Only methods and literal constants may appear in a module class')
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.decorator_list or (node.name.startswith('__') and node.name != '__init__'):
            self.error(node, 'Decorators/special-method overrides are not allowed')
        if node.name in self.aliases or node.name in SAFE_BUILTINS:
            self.error(node, 'Imports and builtins cannot be shadowed by a function')
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith('__'):
            if not (node.attr == '__init__' and (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (node.value.func.id == 'super') or self.path(node.value) == 'torch.nn.Module')):
                self.error(node, 'Dunder attribute/introspection is not allowed')
        else:
            path = self.path(node)
            if path:
                if not self.allowed_path(path):
                    self.error(node, f'Torch/math member is not allowed: {path}')
            elif isinstance(node.value, ast.Name) and node.value.id == 'self':
                if node.attr not in self.self_fields | self.methods | TENSOR_METHODS | TENSOR_FIELDS:
                    self.error(node, f'Undeclared/unsafe self attribute: {node.attr}')
            elif node.attr not in TENSOR_METHODS | TENSOR_FIELDS:
                self.error(node, f'Tensor/module attribute is not allowed: {node.attr}')
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id not in self.bound | SAFE_BUILTINS:
            self.error(node, f'Unsafe/dynamic call: {node.func.id}')
        if isinstance(node.func, ast.Name) and node.func.id == 'super':
            if node.keywords or len(node.args) not in (0, 2):
                self.error(node, 'Unsupported super() call')
            elif node.args and (not (isinstance(node.args[0], ast.Name) and node.args[0].id in self.classes and isinstance(node.args[1], ast.Name) and (node.args[1].id == 'self'))):
                self.error(node, 'super() must refer to a local module class and self')
        if not isinstance(node.func, (ast.Name, ast.Attribute, ast.Subscript)):
            self.error(node, 'Computed callable expressions are not allowed')
        self.generic_visit(node)

def validate_candidate_static(code: str, branch: str) -> dict:
    report = {'accepted': False, 'stage': 'static', 'errors': [], 'code_sha256': None, 'normalized_sha256': None}
    try:
        spec = branch_spec(branch)
        report.update(branch=spec['branch'], class_name=spec['class_name'])
        clean = extract_python_code(code)
        report['code'] = clean
        report['code_sha256'] = hashlib.sha256(clean.encode()).hexdigest()
        tree = ast.parse(clean)
        if sum((1 for _ in ast.walk(tree))) > MAX_AST_NODES:
            raise ValueError('Candidate AST exceeds complexity limit')
        report['normalized_sha256'] = normalized_code_hash(clean)
        report['errors'] = _StaticGuard(tree, spec).check()
        report['accepted'] = not report['errors']
    except (ValueError, TypeError, SyntaxError, RecursionError) as error:
        report['errors'].append(str(error)[:2000])
    return report
