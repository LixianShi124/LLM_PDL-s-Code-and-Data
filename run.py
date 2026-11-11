from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {'generate': 'generation/generate_dataset.py', 'prepare': 'pgenas/prepare.py',
            'search': 'pgenas/run.py', 'prompts': 'pgenas/run.py',
            'train': 'code/train.py', 'evaluate': 'code/evaluate.py'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=COMMANDS)
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        parser.parse_args()
        return
    stage, arguments = sys.argv[1], sys.argv[2:]
    if stage == 'search' and '--smoke' not in arguments:
        arguments = ['--search', *arguments]
    raise SystemExit(subprocess.call([sys.executable, '-B', str(ROOT / COMMANDS[stage]), *arguments]))


if __name__ == '__main__':
    main()
