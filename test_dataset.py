"""Checks of the scientific data contract, independent signals and disk round trips."""

from collections import Counter

import numpy as np
import pytest

import generate_dataset as gd


def test_catalog_exhausts_current_label_structure_without_duplicates():
    catalog = gd.build_catalog()
    assert len(catalog) == 64
    assert len({tuple(r["y"]) for r in catalog}) == 64
    assert Counter(r["multiplicity"] for r in catalog) == {0: 1, 1: 8, 2: 22, 3: 24, 4: 9}
    for record in catalog:
        y = record["y"]
        assert sum(y[i] for i in (0, 1, 4)) <= 1
        assert sum(y[i] for i in (5, 6, 7)) <= 1
    duplicate_old_ids = next(r for r in catalog if r["name"] == "harmonic+interruption+oscillatory")
    assert duplicate_old_ids["legacy_class_ids"] == [25, 41]


def test_composition_splits_obey_holdouts_and_label_balance():
    catalog = gd.build_catalog()
    fingerprints = set()
    for seed in (101, 102, 103):
        split = gd.make_partition(catalog, seed)
        assert split == gd.make_partition(catalog, seed)
        seen, unseen = set(split["seen_class_ids"]), set(split["unseen_class_ids"])
        assert not seen & unseen and seen | unseen == set(range(64))
        assert len(seen) == 37 and len(unseen) == 27
        assert {r["class_id"] for r in catalog if r["multiplicity"] <= 1} <= seen
        assert {r["class_id"] for r in catalog if r["multiplicity"] == 4} <= unseen
        for multiplicity, count in ((2, 6), (3, 12)):
            held = [r for r in catalog if r["class_id"] in unseen and r["multiplicity"] == multiplicity]
            pool = [r for r in catalog if r["multiplicity"] == multiplicity]
            actual = np.asarray([r["y"] for r in held]).sum(axis=0)
            expected = np.asarray([r["y"] for r in pool]).sum(axis=0) * count / len(pool)
            assert np.all(actual >= np.floor(expected))
            assert np.all(actual <= np.ceil(expected))
            assert np.all(actual > 0)
        fingerprints.add(tuple(sorted(unseen)))
    assert len(fingerprints) == 3


def blank_params():
    return {name: 0.0 for name in gd.PARAMETER_COLUMNS}


def test_normal_and_harmonic_signals_match_analytic_spectrum():
    spec = gd.protocol()
    t = np.arange(1200) / 6000
    normal = gd.render_waveform(blank_params(), spec)
    assert np.allclose(normal, np.sin(2 * np.pi * 50 * t), atol=3e-8, rtol=1e-7)
    p = blank_params()
    p.update(harmonic_3_pu=0.10, harmonic_5_pu=0.08, harmonic_7_pu=0.06)
    wave = gd.render_waveform(p, spec)
    spectrum = 2 * np.abs(np.fft.rfft(wave)) / len(wave)
    for frequency, amplitude in ((50, 1.0), (150, 0.1), (250, 0.08), (350, 0.06)):
        assert spectrum[frequency // 5] == pytest.approx(amplitude, abs=1e-7)
    wanted = {10, 30, 50, 70}
    assert max(spectrum[i] for i in range(len(spectrum)) if i not in wanted) < 1e-7


@pytest.mark.parametrize("delta", [0.4, -0.5, -0.95])
def test_amplitude_events_have_correct_levels_and_window_clipping(delta):
    spec = gd.protocol()
    p = blank_params()
    p.update(carrier_phase_rad=0.4, amp_start_s=0.08, amp_duration_s=0.18,
             amp_end_s=0.26, amp_observed_duration_s=0.12, amp_delta_pu=delta)
    t = np.arange(1200) / 6000
    carrier = np.sin(2 * np.pi * 50 * t + 0.4)
    expected = carrier.copy()
    expected[t > 0.08] *= 1 + delta
    assert np.allclose(gd.render_waveform(p, spec), expected, atol=1e-7)


def test_localized_impulsive_and_oscillatory_components_do_not_leak_outside_event():
    spec = gd.protocol()
    base = blank_params()
    base["carrier_phase_rad"] = 0.33
    normal = gd.render_waveform(base, spec)
    t = np.arange(1200) / 6000
    for key, values, start, duration in [
        ("impulse_gain_pu", {"impulse_gain_pu": 5.0, "impulse_start_s": 0.07, "impulse_duration_s": 0.002}, 0.07, 0.002),
        ("osc_amplitude_pu", {"osc_amplitude_pu": 0.7, "osc_start_s": 0.06, "osc_duration_s": 0.03,
                              "osc_frequency_hz": 600.0, "osc_decay_per_s": 60.0}, 0.06, 0.03),
    ]:
        wave = gd.render_waveform({**base, **values}, spec)
        inside = (t > start) & (t <= start + duration)
        assert np.array_equal(wave[~inside], normal[~inside]), key
        assert np.max(np.abs(wave[inside] - normal[inside])) > 0.1, key


def test_notches_remain_periodic_with_original_waveform_preserved_elsewhere():
    spec = gd.protocol()
    p = blank_params()
    p.update(carrier_phase_rad=0.7, notch_base_start_s=0.05, notch_width_s=0.0007, notch_depth_pu=0.3)
    t = np.arange(1200) / 6000
    wave = gd.render_waveform(p, spec)
    pure = np.sin(2 * np.pi * 50 * t + 0.7)
    affected = np.zeros(1200, dtype=bool)
    for n in range(1, 10):
        affected |= (t > 0.05 + 0.02 * n) & (t <= 0.0507 + 0.02 * n)
    assert np.allclose(wave[~affected], pure[~affected], atol=5e-8)
    assert np.allclose(np.abs(wave[affected] - pure[affected]), np.minimum(0.3, np.abs(pure[affected])), atol=1e-7)
    assert np.all(np.abs(wave[affected]) <= np.abs(pure[affected]) + 1e-7)
    assert np.all(wave[affected] * pure[affected] >= -1e-9)


def test_deep_interruption_plus_flicker_preserves_positive_remaining_envelope():
    """Regression: V1 yielded a negative gain for precisely this example."""
    spec = gd.protocol()
    p = blank_params()
    p.update(carrier_phase_rad=0.37, amp_start_s=0.04, amp_duration_s=0.15,
             amp_end_s=0.19, amp_observed_duration_s=0.15, amp_delta_pu=-0.99,
             flicker_depth_pu=0.2, flicker_frequency_hz=5.0, flicker_phase_rad=0.0)
    t = np.arange(1200) / 6000
    carrier = np.sin(2 * np.pi * 50 * t + 0.37)
    active = (t > 0.04) & (t <= 0.19)
    amplitude = np.where(active, 0.01, 1.0)
    gain = amplitude * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))
    wave = gd.render_waveform(p, spec)
    assert gain.min() >= 0.008 - 1e-12
    assert np.allclose(wave, gain * carrier, atol=1e-7)
    assert np.all(wave * carrier >= -1e-9)
    old_gain = 1 + np.where(active, -0.99, 0) + 0.2 * np.sin(2 * np.pi * 5 * t)
    assert np.any(old_gain < 0), "The regression example must exercise the old failure"


def test_compound_flicker_matches_independent_sideband_identity():
    """Use sidebands rather than the generator's envelope multiplication."""
    spec = gd.protocol()
    p = blank_params()
    p.update(carrier_phase_rad=0.73, amp_start_s=0.045, amp_duration_s=0.12,
             amp_end_s=0.165, amp_observed_duration_s=0.12, amp_delta_pu=0.6,
             flicker_depth_pu=0.17, flicker_frequency_hz=12.3, flicker_phase_rad=1.23,
             harmonic_3_pu=0.1, harmonic_5_pu=0.07, harmonic_7_pu=0.09)
    t = np.arange(1200) / 6000
    scale = np.where((t > 0.045) & (t <= 0.165), 1.6, 1.0)
    sidebands = 0.17 / 2 * (
        np.cos(2 * np.pi * (50 - 12.3) * t + 0.73 - 1.23)
        - np.cos(2 * np.pi * (50 + 12.3) * t + 0.73 + 1.23))
    expected = scale * (np.sin(2 * np.pi * 50 * t + 0.73) + sidebands)
    for order, amplitude in ((3, 0.1), (5, 0.07), (7, 0.09)):
        expected += amplitude * np.sin(2 * np.pi * 50 * order * t + order * 0.73)
    assert np.allclose(gd.render_waveform(p, spec), expected, atol=2e-7)


def test_notching_uses_complete_waveform_and_never_amplifies_or_reverses_it():
    """Deep interruption and harmonics force notch saturation and non-carrier polarity."""
    spec = gd.protocol()
    p = blank_params()
    p.update(carrier_phase_rad=0.2, amp_start_s=0.04, amp_duration_s=0.16,
             amp_end_s=0.2, amp_observed_duration_s=0.16, amp_delta_pu=-0.95,
             harmonic_3_pu=0.13, harmonic_5_pu=0.11, harmonic_7_pu=0.07,
             notch_base_start_s=0.05, notch_width_s=0.0008, notch_depth_pu=0.3)
    pre = gd.render_waveform({**p, "notch_depth_pu": 0.0}, spec)
    after = gd.render_waveform(p, spec)
    assert np.all(np.abs(after) <= np.abs(pre) + 1e-7)
    assert np.all(after * pre >= -1e-9)
    assert np.any((after == 0) & (np.abs(pre) > 1e-5))


def test_stream_reproducibility_and_independence_between_roles():
    spec = gd.protocol()
    item = next(r for r in gd.build_catalog() if r["multiplicity"] == 4)
    train, p1 = gd.make_record(item, spec, 20261024, "train", 0)
    repeated, p2 = gd.make_record(item, spec, 20261024, "train", 0)
    test, _ = gd.make_record(item, spec, 20261024, "test_unseen", 0)
    assert np.array_equal(train, repeated) and p1 == p2
    assert not np.array_equal(train, test)


def test_small_full_pipeline_round_trip_and_overwrite_protection(tmp_path):
    destination = tmp_path / "small_dataset"
    counts = {"train": 2, "validation": 1, "test_seen": 3, "test_unseen": 3}
    manifest = gd.generate(destination, counts)
    assert manifest["total_samples"] == 3 * (37 * (2 + 1 + 3) + 27 * 3)
    report = gd.audit(destination)
    assert report["status"] == "passed" and report["metadata_reconstruction_checked_all_rows"]
    with pytest.raises(FileExistsError):
        gd.generate(destination, counts)
