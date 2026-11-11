from pathlib import Path
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from confidence_calibration import BRANCH_NAMES, PAPER_LABEL_NAMES, map_branch_predictions, calibrated_predictions, load_temperatures
from direct_fusion_data import AllDataset
from train import load_training_models, _training_logits

def storage_to_paper(y):
    return np.asarray(y)[:, [0, 1, 4, 2, 3, 5, 6, 7]]


def predict_branches(models, inputs, temperatures):
    logits = {branch: _training_logits(models[branch](inputs[j]), len(inputs[j]), classes)
              for j, (branch, classes) in enumerate(zip(BRANCH_NAMES, (4, 2, 2, 4)))}
    return calibrated_predictions(logits, temperatures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--architecture-dir', type=Path, default=Path(__file__).resolve().parent / 'architectures')
    parser.add_argument('--temperature-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dataset = AllDataset(args.data, 'test')
    if args.data.suffix.lower() == '.npz':
        with np.load(args.data, allow_pickle=False) as saved:
            np.testing.assert_array_equal(dataset.Multi_labels, storage_to_paper(saved['y']))
    models = load_training_models(args.architecture_dir.resolve(), device)
    filenames = ('Causal_Amp.pth', 'Causal_harm.pth', 'Causal_flicker.pth', 'Causal_tran.pth')
    for branch, filename in zip(BRANCH_NAMES, filenames):
        weights = torch.load(args.model_dir / filename, map_location=device, weights_only=True)
        models[branch].load_state_dict(weights, strict=True)
        models[branch].eval()
    temperature_file = args.temperature_file or args.model_dir / 'temperature_scaling.json'
    temperatures = load_temperatures(temperature_file, models)
    predictions, branch_predictions, confidence = ([], [], [])
    probability_arrays = {branch: [] for branch in BRANCH_NAMES}
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            states_by_branch, probabilities, q = predict_branches(models, [batch[2*j].to(device) for j in range(4)], temperatures)
            states = [states_by_branch[branch] for branch in BRANCH_NAMES]
            predictions.append(map_branch_predictions(*states).cpu().numpy())
            branch_predictions.append(torch.stack(states, dim=1).cpu().numpy())
            confidence.append(q.cpu().numpy())
            for branch in BRANCH_NAMES:
                probability_arrays[branch].append(probabilities[branch].cpu().numpy())
    y_true, y_pred = (dataset.Multi_labels, np.concatenate(predictions))
    report = dict(n=len(dataset), label_order=list(PAPER_LABEL_NAMES), exact_match=float(accuracy_score(y_true, y_pred)), micro_f1=float(f1_score(y_true, y_pred, average='micro', zero_division=0)), macro_f1=float(f1_score(y_true, y_pred, average='macro', zero_division=0)), hamming_loss=float(hamming_loss(y_true, y_pred)), decision='calibrated_branch_argmax', temperatures=temperatures)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'predictions.npz', sample_id=dataset.sample_ids, class_id=dataset.class_ids, y_true=y_true, y_pred=y_pred, branch_pred=np.concatenate(branch_predictions), label_order=np.array(PAPER_LABEL_NAMES), branch_confidence=np.concatenate(confidence), branch_order=np.array(BRANCH_NAMES), temperatures=np.array([temperatures[b] for b in BRANCH_NAMES]), **{b+'_probabilities': np.concatenate(values) for b, values in probability_arrays.items()})
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
