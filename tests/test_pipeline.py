"""
Unit tests and pipeline verification for cuffless BP estimation modules.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import neurokit2 as nk

from src.preprocessing.signal_cleaner import preprocess_ppg
from src.features.extractors import (
    extract_prv_dynamics,
    extract_mptp_morphology,
    MORPHOLOGY_FEATURES,
    DYNAMICS_FEATURES,
)
from src.models.cascaded_bpe_net import (
    MorphologyDNN,
    SBPNet,
    DBPNet,
    CascadedBPENet,
)
from src.data.build_dataset import extract_uci_ground_truth_bp, process_window


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.fs = 100
        # 30 seconds of simulated PPG
        self.raw_ppg = nk.ppg_simulate(
            duration=30,
            sampling_rate=self.fs,
            heart_rate=72,
            random_state=42,
        )

    def test_preprocess_ppg_success(self):
        cleaned, peaks, troughs = preprocess_ppg(self.raw_ppg, sampling_rate=self.fs)
        self.assertEqual(len(cleaned), len(self.raw_ppg))
        self.assertGreater(len(peaks), 15)
        self.assertGreater(len(troughs), 15)
        self.assertTrue(np.all(np.isfinite(cleaned)))

    def test_preprocess_ppg_input_validation(self):
        with self.assertRaises(ValueError):
            # Too short (< 1s)
            preprocess_ppg(np.ones(50), sampling_rate=100)


class TestFeatureExtractors(unittest.TestCase):
    def setUp(self):
        self.fs = 100
        # 120 seconds of simulated PPG
        self.raw_ppg = nk.ppg_simulate(
            duration=120,
            sampling_rate=self.fs,
            heart_rate=75,
            random_state=42,
        )
        self.cleaned, self.peaks, self.troughs = preprocess_ppg(
            self.raw_ppg, sampling_rate=self.fs
        )

    def test_extract_mptp_morphology(self):
        morph_series = extract_mptp_morphology(self.cleaned, self.peaks, self.troughs)
        self.assertIsInstance(morph_series, pd.Series)
        self.assertEqual(list(morph_series.index), MORPHOLOGY_FEATURES)
        self.assertFalse(morph_series.isna().any())
        # Check physiological bounds: cardiac period > 0, sum_w > 0
        self.assertGreater(morph_series["cardiac_period"], 40.0)
        self.assertGreater(morph_series["ratio_10"], 0.0)

    def test_extract_prv_dynamics(self):
        dyn_series = extract_prv_dynamics(self.peaks, sampling_rate=self.fs)
        self.assertIsInstance(dyn_series, pd.Series)
        self.assertEqual(list(dyn_series.index), DYNAMICS_FEATURES)
        self.assertFalse(dyn_series.isna().any())
        self.assertGreater(dyn_series["SDNN"], 0.0)


class TestPyTorchModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.morph_x = torch.randn(self.batch_size, 21)
        self.dyn_x = torch.randn(self.batch_size, 7)

    def test_morphology_dnn(self):
        model = MorphologyDNN()
        out = model(self.morph_x)
        self.assertEqual(out.shape, (self.batch_size, 2))

    def test_stage2_nets(self):
        sbp_net = SBPNet()
        dbp_net = DBPNet()
        in_sbp = torch.randn(self.batch_size, 8)
        in_dbp = torch.randn(self.batch_size, 8)

        out_sbp = sbp_net(in_sbp)
        out_dbp = dbp_net(in_dbp)

        self.assertEqual(out_sbp.shape, (self.batch_size, 1))
        self.assertEqual(out_dbp.shape, (self.batch_size, 1))

    def test_cascaded_bpe_net_forward_and_backward(self):
        model = CascadedBPENet()
        f_sbp, f_dbp, p_sbp, p_dbp = model(self.morph_x, self.dyn_x)

        self.assertEqual(f_sbp.shape, (self.batch_size, 1))
        self.assertEqual(f_dbp.shape, (self.batch_size, 1))
        self.assertEqual(p_sbp.shape, (self.batch_size, 1))
        self.assertEqual(p_dbp.shape, (self.batch_size, 1))

        loss = f_sbp.mean() + f_dbp.mean() + p_sbp.mean() + p_dbp.mean()
        loss.backward()

        # Check gradients computed
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient missing for {name}")


class TestDataPipeline(unittest.TestCase):
    def test_abp_ground_truth_extraction(self):
        fs = 125
        t = np.linspace(0, 30, 30 * fs)
        # Synthetic arterial line signal: baseline 80 + 40*pulse
        abp_signal = 80 + 35 * np.sin(2 * np.pi * 1.2 * t)
        sbp, dbp = extract_uci_ground_truth_bp(abp_signal, sampling_rate=fs)
        self.assertTrue(np.isfinite(sbp))
        self.assertTrue(np.isfinite(dbp))
        self.assertGreater(sbp, dbp)


if __name__ == "__main__":
    unittest.main()
