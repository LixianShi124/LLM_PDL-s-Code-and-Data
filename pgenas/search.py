import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from pgenas.candidate_guard import validate_candidate_static
from pgenas.data import load_data, sha256
from pgenas.paper_prompts import SPECS, render_prompt, seed_code
from pgenas.pareto import objectives, select_parent, select_final, update_archive
HERE = Path(__file__).resolve().parent

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def validate_config(config):
    branches = config['branches']
    if not branches or len(set(branches)) != len(branches) or any((b not in SPECS for b in branches)):
        raise ValueError('Choose distinct known branches')
    for key in ('seed', 'generations', 'population', 'archive_capacity', 'epochs', 'batch_size', 'threads', 'candidate_timeout', 'max_api_calls', 'api_timeout', 'api_max_tokens'):
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid integer setting: {key}')
    if config['batch_size'] < 2 or not math.isfinite(config['learning_rate']) or config['learning_rate'] <= 0:
        raise ValueError('Invalid training settings')
    if config['device'] not in ('cpu', 'cuda') or not 0 <= config['temperature'] <= 2:
        raise ValueError('Invalid device or temperature')
    if config['transient_preprocessing'] != 'fundamental_first':
        raise ValueError('transient_preprocessing must be fundamental_first')
    for b in branches:
        if config['budgets'][b] != dict(max_params=50000, deploy_params=50000, max_flops=50000000, deploy_flops=50000000):
            raise ValueError('Per-branch budgets must be 50000 parameters and 50000000 FLOPs')
    requested = len(branches) * config['generations'] * config['population']
    if requested > config['max_api_calls']:
        raise ValueError('Planned proposal slots exceed max_api_calls; reduce the schedule or increase the limit explicitly')

def evaluate_code(code, branch, directory, config, arrays, candidate_id, seen):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'original_response.py').write_text(code, encoding='utf-8')
    checked = validate_candidate_static(code, branch)
    save(directory / 'static_check.json', checked)
    record = dict(id=candidate_id, branch=branch, path=str(directory), status='invalid')
    if not checked['accepted']:
        record['error'] = '; '.join(checked['errors'])
        return record
    fingerprint = checked['normalized_sha256']
    record['code_hash'] = fingerprint
    if fingerprint in seen:
        return dict(record, status='duplicate', error='Duplicate architecture; no retraining')
    seen.add(fingerprint)
    source = directory / 'architecture.py'
    source.write_text(checked['code'], encoding='utf-8')
    job = dict(arrays, **config['budgets'][branch], branch=branch, code_path=str(source), output=str(directory))
    for key in ('seed', 'threads', 'device', 'epochs', 'batch_size', 'learning_rate', 'transient_preprocessing'):
        job[key] = config[key]
    save(directory / 'job.json', job)
    environment = {k: v for k, v in os.environ.items() if not any((s in k.upper() for s in ('API_KEY', 'API_TOKEN', 'OPENAI_API', 'PGENAS_API')))}
    environment.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OMP_NUM_THREADS=str(config['threads']), MKL_NUM_THREADS=str(config['threads']))
    try:
        with (directory / 'training.log').open('w', encoding='utf-8') as log:
            process = subprocess.run([sys.executable, '-B', str(HERE / 'evaluate.py'), '--job', str(directory / 'job.json')], stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=config['candidate_timeout'], cwd=HERE)
        result = json.loads((directory / 'evaluation.json').read_text(encoding='utf-8'))
        if result['status'] == 'trained':
            if process.returncode or result['validation_count'] != arrays['validation_count'] or (not (directory / 'best_validation.pt').exists()):
                raise ValueError('Incomplete candidate evaluation')
        for key in ('status', 'accuracy', 'params', 'flops', 'error', 'elapsed_seconds'):
            if key in result:
                record[key] = result[key]
    except subprocess.TimeoutExpired:
        record.update(status='timeout', error='Candidate evaluation timed out')
    except (OSError, ValueError, KeyError) as error:
        record.update(status='failed', error=str(error))
    return record

def feedback(row):
    return {k: row[k] for k in ('id', 'accuracy', 'params', 'flops')}

def run_search(config, data_dir, output, api):
    validate_config(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing runs are preserved')
    arrays, data_manifest = load_data(data_dir, config['branches'])
    output.mkdir(parents=True)
    started = time.monotonic()
    save(output / 'config.json', config)
    save(output / 'data_manifest.json', data_manifest)
    save(output / 'source_manifest.json', {str(p.relative_to(HERE)): sha256(p) for p in [*HERE.glob('*.py'), *HERE.glob('prompts/*.txt')]})
    rng = random.Random(config['seed'])
    history, selected, memories = ([], {}, {})
    for branch in config['branches']:
        archive, seen, memory = ([], set(), [])
        budget = config['budgets'][branch]

        def record_result(row):
            history.append(row)
            if row['status'] != 'trained':
                memory.append(f'{row['id']}: {row.get('error', row['status'])}'[:2000])
            save(Path(row['path']) / 'record.json', row)
            save(output / 'candidate_history.json', history)
            save(output / branch / 'failure_memory.json', memory)
        initial = evaluate_code(seed_code(branch), branch, output / branch / 'initial', config, arrays[branch], branch + '_initial', seen)
        initial.update(generation=0, mode=None, parent_id=None)
        record_result(initial)
        if initial['status'] == 'trained':
            archive = [initial]
        save(output / branch / 'archive_000.json', archive)
        for generation in range(1, config['generations'] + 1):
            offspring = []
            for slot in range(config['population']):
                mode = 'exploration' if slot % 2 == 0 else 'exploitation'
                parent = select_parent(archive, mode, rng) if archive else None
                parent_code = (Path(parent['path']) / 'architecture.py').read_text(encoding='utf-8') if parent else ''
                prompt = render_prompt(branch, mode, parent_code, feedback(parent) if parent else None, [feedback(row) for row in archive], memory)
                name = f'{branch}_g{generation:02d}_s{slot:02d}'
                directory = output / branch / name
                directory.mkdir(parents=True)
                (directory / 'prompt.txt').write_text(prompt, encoding='utf-8')
                reply = api.generate(prompt)
                if reply['success']:
                    row = evaluate_code(reply['code'], branch, directory, config, arrays[branch], name, seen)
                else:
                    row = dict(id=name, branch=branch, path=str(directory), status='api_failed', error=reply['error'])
                row.update(generation=generation, mode=mode, parent_id=parent['id'] if parent else None, call_index=reply.get('call_index'))
                record_result(row)
                if row['status'] == 'trained':
                    offspring.append(row)
            archive = update_archive(archive, offspring, config['archive_capacity'])
            for row in archive:
                row['objectives'] = objectives(row, budget)
            save(output / branch / f'archive_{generation:03d}.json', archive)
        memories[branch] = memory
        try:
            winner = select_final(archive, budget)
            target = output / 'selected' / branch
            target.mkdir(parents=True)
            for filename in ('architecture.py', 'best_validation.pt'):
                shutil.copy2(Path(winner['path']) / filename, target / filename)
            selected[branch] = dict(winner, selected_path=str(target))
        except ValueError as error:
            selected[branch] = dict(status='no_feasible_candidate', error=str(error))
        save(output / 'selection.json', selected)
    result = dict(status='complete' if all((x.get('status') == 'trained' for x in selected.values())) else 'no_feasible_candidate', selection=selected, api=api.summary(), elapsed_seconds=time.monotonic() - started, test_data_accessed=False)
    save(output / 'result.json', result)
    return result
