import unittest

import numpy as np

from scripts.methods import hard_threshold_columns, run_qdla


class QDLATests(unittest.TestCase):
    def test_top_one_loss_is_nearest_axis_loss(self):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(100, 5))
        q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        coordinates = data @ q
        codes = hard_threshold_columns(coordinates, 1)
        threshold_loss = np.sum((coordinates - codes) ** 2)
        axis_loss = np.sum(np.sum(data ** 2, axis=1) - np.max(coordinates ** 2, axis=1))
        self.assertAlmostEqual(threshold_loss, axis_loss)

    def test_qdla_decreases_loss_and_returns_orthogonal_dictionary(self):
        rng = np.random.default_rng(12)
        data = rng.normal(size=(500, 5))
        initial, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        initial_coordinates = data @ initial
        initial_codes = hard_threshold_columns(initial_coordinates, 1)
        initial_loss = np.sum((initial_coordinates - initial_codes) ** 2)
        fitted, _, success, details = run_qdla(data, 5, initial_point=initial)
        fitted_coordinates = data @ fitted
        fitted_codes = hard_threshold_columns(fitted_coordinates, 1)
        fitted_loss = np.sum((fitted_coordinates - fitted_codes) ** 2)
        self.assertTrue(success)
        self.assertLessEqual(fitted_loss, initial_loss + 1e-10)
        self.assertGreater(details['iterations'], 0)
        np.testing.assert_allclose(fitted.T @ fitted, np.eye(5), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
