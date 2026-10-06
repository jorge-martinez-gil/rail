"""Statistical checks for the release analysis; runnable without pytest."""
import unittest

import numpy as np

from experiments.study_release import cluster_bootstrap, counts, metrics


class StudyReleaseTests(unittest.TestCase):
    def test_confusion_counts_and_retention(self):
        records = [
            {'admitted': 1, 'operator_label': 'A', 'truth': 'A'},
            {'admitted': 1, 'operator_label': 'B', 'truth': 'A'},
            {'admitted': 0, 'operator_label': 'A', 'truth': 'A'},
            {'admitted': 0, 'operator_label': 'B', 'truth': 'A'},
        ]
        np.testing.assert_array_equal(counts(records), [4, 2, 2, 1])
        rates = metrics(counts(records))
        self.assertEqual(rates['yield'], 0.5)
        self.assertEqual(rates['clean_retention'], 0.5)
        self.assertEqual(rates['error_reduction'], 0)

    def test_conditions_are_not_exchanged(self):
        # One whole session per stratum: every resample must retain both,
        # even though the strata have opposite admission patterns.
        result = cluster_bootstrap([[[20, 20, 0, 0]], [[20, 0, 10, 0]]], 100)
        self.assertEqual(result['yield']['ci95'], [0.5, 0.5])
        self.assertEqual(result['base_error']['ci95'], [0.25, 0.25])

    def test_whole_sessions_are_resampled(self):
        # Whole-session sampling reaches all/none admitted; event-level
        # resampling of forty observations would not have these 95% endpoints.
        result = cluster_bootstrap([[[20, 20, 1, 1], [20, 0, 1, 0]]], 2000)
        self.assertEqual(result['yield']['ci95'], [0.0, 1.0])
        self.assertLess(result['admitted_error']['defined_replicates'], 2000)

    def test_zero_errors_and_reproducibility(self):
        result = cluster_bootstrap([[[20, 10, 0, 0]]], 100)
        self.assertIsNone(result['false_admission']['ci95'])
        self.assertEqual(result['false_admission']['defined_replicates'], 0)
        self.assertEqual(result, cluster_bootstrap([[[20, 10, 0, 0]]], 100))


if __name__ == '__main__':
    unittest.main()
