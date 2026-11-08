"""V2 clean PQD data with nonnegative modulation and bounded notching.

All combinations use one explicit waveform model. No model predictions,
weights, or test scores are consulted when sampling or assigning combinations.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "a0_PQD_datageneration" / (
    "a1_data_generation_with_new_disturbance_测试集中多三重扰动+四重扰动.py"
)
LABELS = ("swell", "sag", "harmonic", "flicker", "interruption",
          "impulsive", "oscillatory", "notch")
CHINESE = ("暂升", "暂降", "谐波", "闪变", "中断", "脉冲暂态", "振荡暂态", "缺口")
ROLE_CODE = {"train": 0, "validation": 1, "test_seen": 2, "test_unseen": 3}
DEFAULT_COUNTS = {"train": 400, "validation": 100, "test_seen": 500, "test_unseen": 500}
EXPECTED_COMBINATIONS = {0: 1, 1: 8, 2: 22, 3: 24, 4: 9}
PARAMETER_COLUMNS = (
    "carrier_phase_rad", "amp_start_s", "amp_duration_s", "amp_end_s",
    "amp_observed_duration_s", "amp_delta_pu", "flicker_depth_pu",
    "flicker_frequency_hz", "flicker_phase_rad", "harmonic_3_pu",
    "harmonic_5_pu", "harmonic_7_pu", "impulse_start_s", "impulse_duration_s",
    "impulse_gain_pu", "osc_start_s", "osc_duration_s", "osc_amplitude_pu",
    "osc_frequency_hz", "osc_decay_per_s", "notch_base_start_s",
    "notch_width_s", "notch_depth_pu", "notch_visible_pulse_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def protocol() -> dict:
    """Parameter ranges come from the old universal_pqd, with phases made explicit."""
    return {
        "schema_version": "pqd-revision-clean-v2",
        "protocol_date": "2026-10-25",
        "sampling_hz": 6000, "fundamental_hz": 50, "duration_s": 0.2,
        "samples_per_record": 1200, "nominal_peak_pu": 1.0,
        "noise_model": "none", "snr_db": None,
        "label_order": list(LABELS),
        "branch_label_order": ["amplitude", "harmonic", "flicker", "transient"],
        "branch_states": {"amplitude": ["none", "swell", "sag", "interruption"],
                          "harmonic": ["none", "harmonic"],
                          "flicker": ["none", "flicker"],
                          "transient": ["none", "impulsive", "oscillatory", "notch"]},
        "waveform_seeds": [20261024, 20261025, 20261026],
        "combination_split_seeds": [101, 102, 103],
        "future_training_seeds": [42, 43, 44],
        "counts_per_combination": DEFAULT_COUNTS.copy(),
        "seen_combination_counts_by_multiplicity": {"0": 1, "1": 8, "2": 16, "3": 12, "4": 0},
        "unseen_combination_counts_by_multiplicity": {"0": 0, "1": 0, "2": 6, "3": 12, "4": 9},
        "rng": "NumPy Generator(PCG64), SeedSequence([waveform_seed, role_code, class_id, sample_index, retry])",
        "split_rule": "Within multiplicity 2/3, select 6/12 unseen combinations with each label's held-out count between floor and ceil of its proportional expectation; first valid seeded draw. All single types seen, all quadruples unseen.",
        "ranges": {
            "carrier_phase_rad": [0.0, float(2 * np.pi)],
            "event_start_s": [0.04, 0.10],
            "amp_duration_s": [0.02, 0.18],
            "swell_increment_pu": [0.1, 0.8],
            "sag_reduction_pu": [0.1, 0.9],
            "interruption_reduction_pu": [0.9, 1.0],
            "harmonic_3_pu": [0.05, 0.15],
            "harmonic_5_pu": [0.05, 0.15],
            "harmonic_7_pu": [0.05, 0.15],
            "flicker_depth_pu": [0.1, 0.2],
            "flicker_frequency_hz": [5.0, 20.0],
            "flicker_phase_rad": [0.0, float(2 * np.pi)],
            "impulse_duration_s": [0.001, 0.003],
            "impulse_gain_pu": [1.0, 10.0],
            "osc_duration_s": [0.01, 0.06],
            "osc_amplitude_pu": [0.1, 0.8],
            "osc_frequency_hz": [300.0, 900.0],
            "osc_decay_per_s": [25.0, 125.0],
            "notch_width_s": [0.0002, 0.001],
            "notch_depth_pu": [0.1, 0.4],
        },
        "distribution": "All listed ranges sampled independently and continuously, uniformly on [lower, upper); harmonic phases are h times the carrier phase; one active state per amplitude/transient branch.",
        "event_interval": "start < t <= end; amplitude events may extend past 0.2 s, as in the old generator; requested and observed durations are both recorded",
        "amplitude_flicker_rule": "Fundamental gain is (1+amplitude_gate)*(1+flicker_modulation); flicker is relative to the remaining fundamental voltage. Add carrier-proportional impulse gain separately. This gain is nonnegative for the configured ranges.",
        "notch_rule": "Pulses start at notch_base_start_s + n/50, n=1,...,9. During a pulse replace the complete pre-notch waveform z by sign(z)*max(abs(z)-notch_depth_pu,0). The requested depth is bounded by abs(z), so notches cannot reverse polarity or increase absolute voltage.",
        "composition_equation": "z=((1+amplitude_gate)*(1+flicker_modulation)+impulse_gate)*sin(2*pi*50*t+phase)+harmonics+damped_oscillation; x=z-sign(z)*min(notch_depth_pu,abs(z))*notch_gates",
        "changes_from_v1": [
            "Relative flicker modulation replaces additive amplitude/flicker mixing, eliminating negative fundamental envelopes.",
            "Notch direction uses the complete pre-notch waveform, and effective depth is limited to its absolute value, eliminating polarity reversal and absolute-amplitude growth.",
            "Class catalog, composition splits, per-class counts, parameter ranges and random seeds remain unchanged. Existing V1 files are preserved separately.",
        ],
        "changes_from_legacy": [
            "One universal composition formula for normal, singles and every compound combination.",
            "Use the legacy universal flicker range 5-20 Hz for all combinations (some old single and compound functions used 5-10 Hz).",
            "Use the legacy universal sag reduction range 0.1-0.9 for all combinations; the old dedicated harmonic+sag function used 0.1-0.5.",
            "Explicit uniform carrier phase in radians and independent uniform flicker modulation phase replace the old random time-origin shift.",
            "One NumPy PCG64 random source replaces mixed Python random and numpy.random globals.",
            "New canonical class_id 0-63 replaces legacy category IDs; class_catalog.csv preserves the legacy mapping, including duplicate legacy IDs for one combination.",
        ],
        "scope": "Explicit mathematical synthetic component models, not field data or an EMT network simulation. No claim of standards certification. Labels denote the sampled disturbance components, not strict voltage thresholds for every instantaneous value in a compound waveform. Multiple amplitude states or multiple transient states in one window are outside the current four-branch label structure. The pulse remains a carrier-proportional short gain burst.",
        "evaluation_policy": "Train uses train only; architecture selection, early stopping and tuning use validation only. test_seen and test_unseen are locked final tests; no selection across splits by test score. No models are trained by this generator.",
    }


def legacy_mapping() -> dict[tuple[int, ...], list[int]]:
    if not SOURCE.exists():
        return {}
    mapping: dict[tuple[int, ...], list[int]] = {}
    old_to_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 27: 7}
    for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "CATEGORY_CONFIG" for t in node.targets):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            old_id = ast.literal_eval(key)
            fields = {ast.literal_eval(k): v for k, v in zip(value.keys, value.values)}
            basic = ast.literal_eval(fields["is_basic"])
            components = ([old_id] if old_id else []) if basic else ast.literal_eval(fields["components"])
            indices = tuple(sorted(old_to_index[c] for c in components))
            mapping.setdefault(indices, []).append(old_id)
    return mapping


def build_catalog() -> list[dict]:
    records = []
    old_ids = legacy_mapping()
    for amplitude, harmonic, flicker, transient in itertools.product(range(4), range(2), range(2), range(4)):
        y = [0] * 8
        if amplitude:
            y[{1: 0, 2: 1, 3: 4}[amplitude]] = 1
        y[2], y[3] = harmonic, flicker
        if transient:
            y[{1: 5, 2: 6, 3: 7}[transient]] = 1
        active = tuple(index for index, bit in enumerate(y) if bit)
        records.append({"indices": active, "y": y,
                        "branch_labels": [amplitude, harmonic, flicker, transient],
                        "name": "+".join(LABELS[i] for i in active) or "normal",
                        "name_zh": "+".join(CHINESE[i] for i in active) or "正常",
                        "multiplicity": len(active), "label_bits": "".join(map(str, y)),
                        "legacy_class_ids": old_ids.get(active, [])})
    records.sort(key=lambda r: (r["multiplicity"], r["indices"]))
    for class_id, item in enumerate(records):
        item["class_id"] = class_id
        item["code"] = f"C{class_id:02d}"
    assert dict(Counter(r["multiplicity"] for r in records)) == EXPECTED_COMBINATIONS
    return records


def balanced_holdout(catalog: list[dict], multiplicity: int, count: int,
                     rng: np.random.Generator) -> tuple[list[int], dict]:
    candidates = [r for r in catalog if r["multiplicity"] == multiplicity]
    matrix = np.asarray([r["y"] for r in candidates], dtype=np.int64)
    expected = matrix.sum(axis=0) * count / len(candidates)
    lower, upper = np.floor(expected).astype(int), np.ceil(expected).astype(int)
    for attempt in range(1, 100_001):
        chosen = rng.choice(len(candidates), size=count, replace=False)
        actual = matrix[chosen].sum(axis=0)
        if np.all(actual >= lower) and np.all(actual <= upper):
            return sorted(candidates[int(i)]["class_id"] for i in chosen), {
                "attempts": attempt, "label_counts": dict(zip(LABELS, map(int, actual))),
                "allowed_label_counts_min": lower.tolist(), "allowed_label_counts_max": upper.tolist(),
            }
    raise RuntimeError("No balanced combination partition found within the predefined draw budget")


def make_partition(catalog: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pairs, pair_balance = balanced_holdout(catalog, 2, 6, rng)
    triples, triple_balance = balanced_holdout(catalog, 3, 12, rng)
    quadruples = [r["class_id"] for r in catalog if r["multiplicity"] == 4]
    unseen = sorted(pairs + triples + quadruples)
    seen = sorted(set(range(64)) - set(unseen))
    return {"combination_split_seed": seed, "seen_class_ids": seen, "unseen_class_ids": unseen,
            "unseen_double_class_ids": pairs, "unseen_triple_class_ids": triples,
            "unseen_quadruple_class_ids": quadruples,
            "balance_diagnostics": {"double": pair_balance, "triple": triple_balance}}


def sample_parameters(item: dict, rng: np.random.Generator, spec: dict) -> dict:
    ranges = spec["ranges"]
    draw = lambda key: float(rng.uniform(*ranges[key]))
    params = {key: 0.0 for key in PARAMETER_COLUMNS}
    params["carrier_phase_rad"] = draw("carrier_phase_rad")
    amplitude, harmonic, flicker, transient = item["branch_labels"]
    if amplitude:
        start, duration = draw("event_start_s"), draw("amp_duration_s")
        key = {1: "swell_increment_pu", 2: "sag_reduction_pu", 3: "interruption_reduction_pu"}[amplitude]
        params.update(amp_start_s=start, amp_duration_s=duration, amp_end_s=start + duration,
                      amp_observed_duration_s=min(start + duration, spec["duration_s"]) - start,
                      amp_delta_pu=draw(key) * (1 if amplitude == 1 else -1))
    if harmonic:
        for order in (3, 5, 7):
            params[f"harmonic_{order}_pu"] = draw(f"harmonic_{order}_pu")
    if flicker:
        for key in ("flicker_depth_pu", "flicker_frequency_hz", "flicker_phase_rad"):
            params[key] = draw(key)
    if transient == 1:
        params.update(impulse_start_s=draw("event_start_s"),
                      impulse_duration_s=draw("impulse_duration_s"), impulse_gain_pu=draw("impulse_gain_pu"))
    elif transient == 2:
        params["osc_start_s"] = draw("event_start_s")
        for key in ("osc_duration_s", "osc_amplitude_pu", "osc_frequency_hz", "osc_decay_per_s"):
            params[key] = draw(key)
    elif transient == 3:
        params.update(notch_base_start_s=draw("event_start_s"),
                      notch_width_s=draw("notch_width_s"), notch_depth_pu=draw("notch_depth_pu"))
        starts = params["notch_base_start_s"] + np.arange(1, 10) / spec["fundamental_hz"]
        params["notch_visible_pulse_count"] = int(np.sum(starts < (spec["samples_per_record"] - 1) / spec["sampling_hz"]))
    return params


def render_waveform(params: dict, spec: dict) -> np.ndarray:
    t = np.arange(spec["samples_per_record"], dtype=np.float64) / spec["sampling_hz"]
    angle = 2 * np.pi * spec["fundamental_hz"] * t + params["carrier_phase_rad"]
    carrier = np.sin(angle)
    envelope = np.ones_like(t)
    if params["amp_delta_pu"]:
        gate = (t > params["amp_start_s"]) & (t <= params["amp_end_s"])
        envelope += params["amp_delta_pu"] * gate
    if params["flicker_depth_pu"]:
        envelope *= 1 + params["flicker_depth_pu"] * np.sin(
            2 * np.pi * params["flicker_frequency_hz"] * t + params["flicker_phase_rad"])
    if params["impulse_gain_pu"]:
        gate = ((t > params["impulse_start_s"]) &
                (t <= params["impulse_start_s"] + params["impulse_duration_s"]))
        envelope += params["impulse_gain_pu"] * gate
    wave = envelope * carrier
    for order in (3, 5, 7):
        wave += params[f"harmonic_{order}_pu"] * np.sin(order * angle)
    if params["osc_amplitude_pu"]:
        elapsed = t - params["osc_start_s"]
        gate = (elapsed > 0) & (elapsed <= params["osc_duration_s"])
        wave[gate] += (params["osc_amplitude_pu"] * np.exp(-params["osc_decay_per_s"] * elapsed[gate])
                       * np.sin(2 * np.pi * params["osc_frequency_hz"] * elapsed[gate]))
    if params["notch_depth_pu"]:
        notch_gate = np.zeros(len(t), dtype=np.float64)
        for n in range(1, 10):
            start = params["notch_base_start_s"] + n / spec["fundamental_hz"]
            notch_gate += (t > start) & (t <= start + params["notch_width_s"])
        wave -= np.sign(wave) * np.minimum(params["notch_depth_pu"], np.abs(wave)) * notch_gate
    return wave.astype("<f4")


def waveform_hashes(wave: np.ndarray) -> tuple[str, str]:
    exact = hashlib.sha256(wave.astype("<f4", copy=False).tobytes()).hexdigest()
    rounded = np.round(wave.astype(np.float64), 6).astype("<f4")
    rounded[rounded == 0] = 0  # Canonicalize signed zero before hashing.
    near = hashlib.sha256(rounded.tobytes()).hexdigest()
    return exact, near


def check_parameter_table(metadata: pd.DataFrame, labels: np.ndarray,
                          branches: np.ndarray, spec: dict) -> None:
    """Check sampled ranges and absence/presence against labels, independently of rendering."""
    def bounded(column: str, key: str, active: np.ndarray) -> None:
        values = metadata[column].to_numpy(dtype=np.float64)
        low, high = spec["ranges"][key]
        assert np.all(np.isfinite(values)), column
        assert np.all(values[active] >= low) and np.all(values[active] < high), column
        assert np.all(values[~active] == 0), f"Inactive parameter is nonzero: {column}"

    all_rows = np.ones(len(metadata), dtype=bool)
    bounded("carrier_phase_rad", "carrier_phase_rad", all_rows)
    amplitude = branches[:, 0] != 0
    bounded("amp_start_s", "event_start_s", amplitude)
    bounded("amp_duration_s", "amp_duration_s", amplitude)
    delta = metadata["amp_delta_pu"].to_numpy()
    for state, key in ((1, "swell_increment_pu"), (2, "sag_reduction_pu"), (3, "interruption_reduction_pu")):
        values = delta[branches[:, 0] == state] * (1 if state == 1 else -1)
        low, high = spec["ranges"][key]
        assert np.all(values >= low) and np.all(values < high), key
    assert np.all(delta[~amplitude] == 0)
    start = metadata["amp_start_s"].to_numpy()
    duration = metadata["amp_duration_s"].to_numpy()
    assert np.allclose(metadata["amp_end_s"], start + duration, rtol=0, atol=1e-15)
    expected_observed = np.minimum(start + duration, spec["duration_s"]) - start
    assert np.allclose(metadata["amp_observed_duration_s"], expected_observed, rtol=0, atol=1e-15)
    for order in (3, 5, 7):
        bounded(f"harmonic_{order}_pu", f"harmonic_{order}_pu", labels[:, 2] != 0)
    for key in ("flicker_depth_pu", "flicker_frequency_hz", "flicker_phase_rad"):
        bounded(key, key, labels[:, 3] != 0)
    bounded("impulse_start_s", "event_start_s", labels[:, 5] != 0)
    for key in ("impulse_duration_s", "impulse_gain_pu"):
        bounded(key, key, labels[:, 5] != 0)
    bounded("osc_start_s", "event_start_s", labels[:, 6] != 0)
    for key in ("osc_duration_s", "osc_amplitude_pu", "osc_frequency_hz", "osc_decay_per_s"):
        bounded(key, key, labels[:, 6] != 0)
    bounded("notch_base_start_s", "event_start_s", labels[:, 7] != 0)
    for key in ("notch_width_s", "notch_depth_pu"):
        bounded(key, key, labels[:, 7] != 0)
    pulse_counts = metadata["notch_visible_pulse_count"].to_numpy()
    assert np.all((pulse_counts[labels[:, 7] != 0] >= 4) & (pulse_counts[labels[:, 7] != 0] <= 7))
    assert np.all(pulse_counts[labels[:, 7] == 0] == 0)


def make_record(item: dict, spec: dict, waveform_seed: int, role: str, sample_index: int,
                retry: int = 0) -> tuple[np.ndarray, dict]:
    words = [waveform_seed, ROLE_CODE[role], item["class_id"], sample_index, retry]
    rng = np.random.default_rng(np.random.SeedSequence(words))
    params = sample_parameters(item, rng, spec)
    return render_waveform(params, spec), params


def write_catalog(catalog: list[dict], output: Path) -> None:
    rows = []
    for item in catalog:
        row = {k: item[k] for k in ("class_id", "code", "name", "name_zh", "multiplicity", "label_bits")}
        row["legacy_class_ids"] = ";".join(map(str, item["legacy_class_ids"]))
        row.update(dict(zip(LABELS, item["y"])))
        row.update(dict(zip(("amplitude_label", "harmonic_label", "flicker_label", "transient_label"), item["branch_labels"])))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "class_catalog.csv", index=False, encoding="utf-8-sig")
    parameter_rows = []
    for name, bounds in protocol()["ranges"].items():
        parameter_rows.append({"parameter": name, "lower": bounds[0], "upper": bounds[1],
                               "distribution": "continuous uniform [lower, upper)"})
    pd.DataFrame(parameter_rows).to_csv(output / "generation_parameters.csv", index=False)


def generate_role(output: Path, item_by_id: dict[int, dict], class_ids: list[int], role: str,
                  count: int, waveform_seed: int, spec: dict, exact_seen: set[str], near_seen: set[str]) -> dict:
    n = len(class_ids) * count
    X = np.empty((n, spec["samples_per_record"]), dtype=np.float32)
    y = np.empty((n, 8), dtype=np.uint8)
    branches = np.empty((n, 4), dtype=np.uint8)
    ids = np.empty(n, dtype=np.int16)
    metadata = []
    retries = 0
    row = 0
    for class_id in class_ids:
        item = item_by_id[class_id]
        for sample_index in range(count):
            for retry in range(100):
                wave, params = make_record(item, spec, waveform_seed, role, sample_index, retry)
                exact, near = waveform_hashes(wave)
                if exact not in exact_seen and near not in near_seen:
                    break
            else:
                raise RuntimeError("Cannot generate an independent waveform after 100 duplicate rejections")
            exact_seen.add(exact)
            near_seen.add(near)
            retries += retry
            X[row], y[row], branches[row], ids[row] = wave, item["y"], item["branch_labels"], class_id
            sample_id = f"W{waveform_seed}-{role}-C{class_id:02d}-{sample_index:04d}"
            metadata.append({"sample_id": sample_id, "role": role, "class_id": class_id,
                             "combination": item["name"], "multiplicity": item["multiplicity"],
                             "waveform_seed": waveform_seed, "role_seed_code": ROLE_CODE[role],
                             "sample_index": sample_index, "retry": retry, "waveform_sha256": exact,
                             "rounded_6dp_sha256": near, **params})
            row += 1
    # Independent row shuffling prevents class-order artifacts; class identities are unchanged.
    order = np.random.default_rng(np.random.SeedSequence([waveform_seed, 900, ROLE_CODE[role]])).permutation(n)
    X, y, branches, ids = X[order], y[order], branches[order], ids[order]
    metadata = [metadata[int(i)] for i in order]
    for row_index, record in enumerate(metadata):
        record["row_index"] = row_index
    sample_ids = np.asarray([record["sample_id"] for record in metadata])
    base = output / f"{role}_clean"
    npz_path, csv_path, metadata_path = base.with_suffix(".npz"), base.with_suffix(".csv"), output / f"{role}_metadata.csv"
    np.savez_compressed(npz_path, X=X, y=y, branch_labels=branches, class_id=ids, sample_id=sample_ids)
    matrix = np.column_stack([X, y, branches, ids]).astype(np.float32, copy=False)
    np.savetxt(csv_path, matrix, delimiter=",", header=",".join(map(str, range(1213))),
               comments="", fmt=["%.9g"] * 1200 + ["%d"] * 13)
    pd.DataFrame(metadata).to_csv(metadata_path, index=False, float_format="%.17g")
    files = {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
             for path in (csv_path, npz_path, metadata_path)}
    return {"role": role, "sample_count": n, "combination_count": len(class_ids),
            "samples_per_combination": count, "class_ids": class_ids,
            "multiplicity_counts": {str(k): v for k, v in sorted(Counter(map(int, y.sum(axis=1))).items())},
            "positive_label_counts": dict(zip(LABELS, map(int, y.sum(axis=0)))),
            "duplicate_rejections": retries, "waveform_abs_max": float(np.max(np.abs(X))),
            "files": files}


def generate(output: Path, counts: dict[str, int] | None = None) -> dict:
    spec = protocol()
    if counts is not None:
        spec["counts_per_combination"] = counts.copy()
    counts = spec["counts_per_combination"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog()
    item_by_id = {item["class_id"]: item for item in catalog}
    write_json(output / "protocol.json", spec)
    write_catalog(catalog, output)
    exact_seen: set[str] = set()
    near_seen: set[str] = set()
    splits = []
    assignment_rows = []
    for split_index, (waveform_seed, split_seed) in enumerate(
            zip(spec["waveform_seeds"], spec["combination_split_seeds"]), 1):
        split_name = f"split_{split_index:02d}"
        split_dir = output / split_name
        split_dir.mkdir()
        partition = make_partition(catalog, split_seed)
        partition.update(split_name=split_name, waveform_seed=waveform_seed)
        write_json(split_dir / "combination_split.json", partition)
        for item in catalog:
            seen = item["class_id"] in partition["seen_class_ids"]
            assignment_rows.append({"split": split_name, "combination_split_seed": split_seed,
                                    "waveform_seed": waveform_seed, "class_id": item["class_id"],
                                    "name": item["name"], "name_zh": item["name_zh"],
                                    "multiplicity": item["multiplicity"], "status": "seen" if seen else "unseen",
                                    "train_count": counts["train"] if seen else 0,
                                    "validation_count": counts["validation"] if seen else 0,
                                    "test_seen_count": counts["test_seen"] if seen else 0,
                                    "test_unseen_count": counts["test_unseen"] if not seen else 0})
        roles = {}
        for role in ROLE_CODE:
            class_ids = partition["unseen_class_ids"] if role == "test_unseen" else partition["seen_class_ids"]
            print(f"[{split_name}] generating {role}: {len(class_ids)} combinations x {counts[role]}", flush=True)
            roles[role] = generate_role(split_dir, item_by_id, class_ids, role, counts[role],
                                        waveform_seed, spec, exact_seen, near_seen)
            print(f"[{split_name}] saved {role}: {roles[role]['sample_count']:,} rows", flush=True)
        split_manifest = {"split_name": split_name, "waveform_seed": waveform_seed,
                          "combination_split_seed": split_seed, "sample_count": sum(v["sample_count"] for v in roles.values()),
                          "roles": roles}
        write_json(split_dir / "manifest.json", split_manifest)
        splits.append(split_manifest)
    pd.DataFrame(assignment_rows).to_csv(output / "combination_assignments.csv", index=False, encoding="utf-8-sig")
    source_hashes = {"generator": sha256_file(Path(__file__).resolve())}
    if SOURCE.exists():
        source_hashes["legacy_reference"] = sha256_file(SOURCE)
    manifest = {"schema_version": spec["schema_version"], "created_utc": datetime.now(timezone.utc).isoformat(),
                "total_samples": sum(s["sample_count"] for s in splits), "split_count": len(splits),
                "unique_exact_waveforms": len(exact_seen), "unique_rounded_6dp_waveforms": len(near_seen),
                "combination_count": 64, "source_sha256": source_hashes,
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "supporting_files": {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
                                     for name in ("protocol.json", "class_catalog.csv", "generation_parameters.csv", "combination_assignments.csv")},
                "splits": splits}
    write_json(output / "manifest.json", manifest)
    print(f"Generation complete: {manifest['total_samples']:,} independent waveforms", flush=True)
    return manifest


def audit(output: Path, check_csv: bool = True) -> dict:
    """Independently reload all artifacts, compare CSV/NPZ and test split isolation."""
    spec = json.loads((output / "protocol.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert sha256_file(Path(__file__).resolve()) == manifest["source_sha256"]["generator"], "Generator has changed since generation; preserve or record a new version"
    catalog = build_catalog()
    lookup_y = np.asarray([item["y"] for item in catalog], dtype=np.uint8)
    lookup_branches = np.asarray([item["branch_labels"] for item in catalog], dtype=np.uint8)
    exact_seen: set[str] = set()
    near_seen: set[str] = set()
    sample_seen: set[str] = set()
    reports = []
    fingerprints = set()
    for filename, info in manifest["supporting_files"].items():
        assert sha256_file(output / filename) == info["sha256"], filename
    for split in manifest["splits"]:
        split_dir = output / split["split_name"]
        partition = json.loads((split_dir / "combination_split.json").read_text())
        assert partition == {**make_partition(catalog, split["combination_split_seed"]),
                             "split_name": split["split_name"], "waveform_seed": split["waveform_seed"]}
        seen, unseen = set(partition["seen_class_ids"]), set(partition["unseen_class_ids"])
        assert len(seen) == 37 and len(unseen) == 27 and not seen & unseen and seen | unseen == set(range(64))
        fingerprint = tuple(sorted(unseen))
        assert fingerprint not in fingerprints, "Composition splits must differ"
        fingerprints.add(fingerprint)
        assert Counter(catalog[i]["multiplicity"] for i in seen) == Counter({0: 1, 1: 8, 2: 16, 3: 12})
        assert Counter(catalog[i]["multiplicity"] for i in unseen) == Counter({2: 6, 3: 12, 4: 9})
        for role, role_info in split["roles"].items():
            print(f"Auditing {split['split_name']}/{role}", flush=True)
            for filename, info in role_info["files"].items():
                assert sha256_file(split_dir / filename) == info["sha256"], filename
            with np.load(split_dir / f"{role}_clean.npz", allow_pickle=False) as saved:
                X, y, branches, ids, sample_ids = (saved[k] for k in ("X", "y", "branch_labels", "class_id", "sample_id"))
            n = role_info["sample_count"]
            assert X.shape == (n, 1200) and X.dtype == np.float32 and np.isfinite(X).all()
            assert np.array_equal(y, lookup_y[ids]) and np.array_equal(branches, lookup_branches[ids])
            expected_ids = unseen if role == "test_unseen" else seen
            assert set(map(int, ids)) == expected_ids
            assert set(Counter(map(int, ids)).values()) == {spec["counts_per_combination"][role]}
            if check_csv:
                csv_data = pd.read_csv(split_dir / f"{role}_clean.csv", dtype=np.float32)
                assert csv_data.shape == (n, 1213) and list(csv_data.columns) == list(map(str, range(1213)))
                assert np.array_equal(csv_data.iloc[:, :1200].to_numpy(), X), "CSV must preserve float32 waveforms exactly"
                assert np.array_equal(csv_data.iloc[:, 1200:1208].to_numpy(), y)
                assert np.array_equal(csv_data.iloc[:, 1208:1212].to_numpy(), branches)
                assert np.array_equal(csv_data.iloc[:, 1212].to_numpy(), ids)
                del csv_data
            metadata = pd.read_csv(split_dir / f"{role}_metadata.csv", float_precision="round_trip")
            assert metadata.shape[0] == n and np.array_equal(metadata["row_index"].to_numpy(), np.arange(n))
            assert np.array_equal(metadata["class_id"].to_numpy(), ids)
            assert np.array_equal(metadata["sample_id"].to_numpy(), sample_ids)
            assert np.all(metadata["waveform_seed"] == split["waveform_seed"])
            assert np.all(metadata["role_seed_code"] == ROLE_CODE[role])
            assert np.all(metadata["role"] == role)
            check_parameter_table(metadata, y, branches, spec)
            # Every row's physical parameters are independently rendered; random
            # stream regeneration is additionally checked for every class in each role.
            for i, record in enumerate(metadata.to_dict("records")):
                wave = X[i]
                exact, near = waveform_hashes(wave)
                assert exact == record["waveform_sha256"] and near == record["rounded_6dp_sha256"]
                assert exact not in exact_seen and near not in near_seen, "Duplicate or rounded duplicate across datasets"
                assert record["sample_id"] not in sample_seen
                exact_seen.add(exact); near_seen.add(near); sample_seen.add(record["sample_id"])
                assert np.array_equal(render_waveform(record, spec), wave), "Metadata must reconstruct each waveform exactly"
            for class_id in expected_ids:
                sample_indices = metadata.loc[ids == class_id, "sample_index"].to_numpy()
                assert np.array_equal(np.sort(sample_indices), np.arange(spec["counts_per_combination"][role])), "Sample indices must be complete and unique within every class/role"
                i = int(np.flatnonzero(ids == class_id)[0])
                row = metadata.iloc[i]
                regenerated, _ = make_record(catalog[class_id], spec, split["waveform_seed"], role,
                                               int(row["sample_index"]), int(row["retry"]))
                assert np.array_equal(regenerated, X[i]), "Stored random stream does not reproduce its waveform"
            reports.append({"split": split["split_name"], "role": role, "rows": n,
                            "class_count": len(expected_ids), "csv_npz_exact_match": bool(check_csv),
                            "all_parameters_reconstruct_waveforms": True,
                            "random_stream_checks": len(expected_ids)})
    assert len(exact_seen) == len(near_seen) == manifest["total_samples"]
    assert manifest["unique_exact_waveforms"] == manifest["unique_rounded_6dp_waveforms"] == len(exact_seen)
    result = {"status": "passed", "total_samples": len(exact_seen), "split_count": len(manifest["splits"]),
              "duplicate_exact_waveforms": 0, "duplicate_rounded_6dp_waveforms": 0,
              "duplicate_sample_ids": 0, "combination_overlap_seen_unseen": 0,
              "csv_npz_comparison": bool(check_csv), "metadata_reconstruction_checked_all_rows": True,
              "counts_labels_hashes_and_seed_reproduction_checked": True, "roles": reports}
    write_json(output / "audit_report.json", result)
    print(f"Audit passed: {len(exact_seen):,} waveforms", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use 2/1/3/3 records per combination in a separate output folder")
    parser.add_argument("--skip-csv-audit", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        counts = {"train": 2, "validation": 1, "test_seen": 3, "test_unseen": 3} if args.smoke else None
        if args.smoke and args.output.resolve() == (ROOT / "data").resolve():
            raise ValueError("Smoke data must use a separate --output directory")
        generate(args.output, counts)
    audit(args.output, check_csv=not args.skip_csv_audit)


if __name__ == "__main__":
    main()
