import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import tempfile
from pgenas.paper_prompts import SPECS, render_prompt, seed_code
HERE = Path(__file__).resolve().parent

class DemoAPI:

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        branch = next((b for b, s in SPECS.items() if f'Required network class: {s['public_class']}' in prompt))
        code = seed_code(branch)
        channels = 1 if branch == 'harm' else SPECS[branch]['shape'][0]
        code = code.replace(f'Conv1d({channels}, 8,', f'Conv1d({channels}, 12,')
        code = code.replace('Linear(12,' if branch == 'harm' else 'Linear(8,', 'Linear(16,' if branch == 'harm' else 'Linear(12,')
        return dict(success=True, code=code, call_index=self.calls)

    def summary(self):
        return dict(requests=0, toy_proposals=self.calls, offline=True, total_tokens=0)

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--search', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if not (args.search or args.smoke):
        output = args.output or HERE / 'prompt_previews'
        output.mkdir(parents=True, exist_ok=True)
        for branch in SPECS:
            (output / f'{branch}.txt').write_text(render_prompt(branch, 'exploration'), encoding='utf-8')
        return
    if args.output is None or args.output.exists():
        parser.error('Specify --output as a new directory')
    from pgenas.data import make_demo_data
    from pgenas.search import run_search, validate_config
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.device:
        config['device'] = args.device
    if args.smoke:
        config.update(device='cpu', generations=1, population=1, epochs=1, batch_size=8, threads=1, max_api_calls=len(config['branches']))
        if args.data_dir is not None:
            result = run_search(config, args.data_dir, args.output, DemoAPI())
        else:
            with tempfile.TemporaryDirectory(prefix='pgenas_demo_') as directory:
                make_demo_data(directory)
                result = run_search(config, directory, args.output, DemoAPI())
    else:
        if args.data_dir is None:
            parser.error('--search requires --data-dir with preprocessed train/validation arrays')
        validate_config(config)
        from pgenas.api_client import APIClient
        api_output = args.output.with_name(args.output.name + '_api')
        if api_output.exists():
            parser.error('API log directory already exists; choose a new output name')
        api = APIClient.from_environment(api_output, config)
        result = run_search(config, args.data_dir, args.output, api)
    if result['status'] != 'complete':
        raise SystemExit(1)
if __name__ == '__main__':
    main()
