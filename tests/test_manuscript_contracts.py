import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.manuscript_source_paths import activate_archive_sources
activate_archive_sources(__file__)
from common.manuscript_cohorts import split_center_a_indices, validate_cohort, validate_base_cohorts
from common.manuscript_bootstrap import BOOTSTRAP_N, shared_bootstrap_indices, bootstrap_identity
import model_factory


def cohort(n, positive, prefix):
    return pd.DataFrame({"record_id": [f"{prefix}{i:04d}" for i in range(n)],
                         "y_SCLC": np.r_[np.ones(positive, dtype=int), np.zeros(n-positive, dtype=int)]})


class CohortTests(unittest.TestCase):
    def test_random_ratio_split_is_reproducible_and_disjoint(self):
        frame = cohort(988, 193, "A")
        train, test = split_center_a_indices(frame)
        self.assertGreaterEqual(len(train) / len(frame), 0.70)
        self.assertLessEqual(len(train) / len(frame), 0.80)
        self.assertEqual(len(train) + len(test), len(frame))
        self.assertLess(abs(frame.iloc[train].y_SCLC.mean() - frame.y_SCLC.mean()), 1 / len(train))
        self.assertEqual((len(train), len(test)), (775, 213))
        self.assertFalse(set(train) & set(test))
        shuffled = frame.sample(frac=1, random_state=7).reset_index(drop=True)
        other_train, other_test = split_center_a_indices(shuffled)
        self.assertEqual(set(frame.iloc[train].record_id), set(shuffled.iloc[other_train].record_id))
        self.assertEqual(set(frame.iloc[test].record_id), set(shuffled.iloc[other_test].record_id))

    def test_uses_actual_cohort_size_and_rejects_duplicate_patients(self):
        for n, events in [(969, 189), (988, 193), (100, 20)]:
            train, test = split_center_a_indices(cohort(n, events, "A"))
            self.assertGreaterEqual(len(train) / n, 0.70)
            self.assertLessEqual(len(train) / n, 0.80)
            self.assertEqual(len(train) + len(test), n)
        frame = cohort(988, 193, "A")
        frame.loc[1, "record_id"] = frame.loc[0, "record_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            split_center_a_indices(frame)

    def test_rejects_wrong_labels_and_cross_center_overlap(self):
        frame = cohort(988, 193, "A")
        frame.loc[0, "y_SCLC"] = 2
        with self.assertRaises(ValueError):
            validate_cohort(frame, "A")
        frames = {"A": cohort(988, 193, "A"), "B": cohort(329, 67, "B"), "C": cohort(313, 70, "C")}
        frames["C"].loc[0, "record_id"] = frames["A"].loc[0, "record_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_base_cohorts(frames)


class BootstrapTests(unittest.TestCase):
    def test_all_2000_draws_preserve_class_counts_and_case_identity(self):
        frame = cohort(31, 7, "test")
        shuffled = frame.sample(frac=1, random_state=13).reset_index(drop=True)
        a = shared_bootstrap_indices(frame.y_SCLC, "A_holdout_clean", frame.record_id)
        b = shared_bootstrap_indices(shuffled.y_SCLC, "A_holdout_clean", shuffled.record_id)
        self.assertEqual(len(a), BOOTSTRAP_N)
        self.assertEqual(len(a), 2000)
        for left, right in zip(a, b):
            self.assertEqual(int(frame.y_SCLC.to_numpy()[left].sum()), 7)
            np.testing.assert_array_equal(frame.record_id.to_numpy()[left], shuffled.record_id.to_numpy()[right])
        self.assertEqual(bootstrap_identity(frame.y_SCLC, frame.record_id),
                         bootstrap_identity(shuffled.y_SCLC, shuffled.record_id))

    def test_rejects_invalid_case_alignment(self):
        with self.assertRaises(ValueError):
            shared_bootstrap_indices([0, 1], "A_holdout_clean", ["same", "same"])
        with self.assertRaises(ValueError):
            shared_bootstrap_indices([0, 0], "A_holdout_clean", ["a", "b"])


class FrozenGAMTests(unittest.TestCase):
    def test_training_and_deployment_factories_never_retry_other_penalties(self):
        deployment_path = ROOT / "04_web_deployment/streamlit_app/runtime/model_factory.py"
        spec = importlib.util.spec_from_file_location("deployment_factory_contract_test", deployment_path)
        deployment_factory = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = deployment_factory
        spec.loader.exec_module(deployment_factory)
        for factory in (model_factory, deployment_factory):
            penalties = []
            class BrokenGAM:
                def __init__(self, **kwargs):
                    penalties.append(kwargs["lam"])
                def fit(self, x, y):
                    raise ArithmeticError("synthetic convergence failure")
            fake_pygam = types.SimpleNamespace(LogisticGAM=BrokenGAM, s=lambda *args, **kwargs: 0)
            with patch.dict(sys.modules, {"pygam": fake_pygam}):
                classifier = factory.PygamLogisticClassifier(lam=0.6)
                with self.assertRaisesRegex(RuntimeError, "frozen lam=0.6"):
                    classifier.fit(np.array([[0., 1.], [1., 2.], [2., 0.], [3., 1.]]), [0, 1, 0, 1])
            self.assertEqual(penalties, [0.6])
            self.assertFalse(hasattr(classifier, "_gam"))


class CalibrationTests(unittest.TestCase):
    def test_unpenalized_calibration_and_equal_width_ece(self):
        from common.manuscript_calibration import calibration_stats
        p = np.array([0.1, 0.1, 0.3, 0.3, 0.6, 0.6, 0.8, 0.8])
        y = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        result = calibration_stats(y, p)
        self.assertTrue(np.isfinite(list(result.values())).all())
        expected_ece = (0.4 + 0.3 + 0.1 + 0.2) / 4
        self.assertAlmostEqual(result["ece_10bin"], expected_ece)
        with self.assertRaises(ValueError):
            calibration_stats(y, np.full(len(y), 0.5))


if __name__ == "__main__":
    unittest.main()
