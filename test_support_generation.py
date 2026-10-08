import unittest

import numpy as np

from support_generation import cocircularity_coeff, gamma, is_cocircular


class CocircularityTests(unittest.TestCase):
    def test_signed_interior_angle(self):
        cases = [
            (0, np.pi / 4, np.pi / 4),
            (0, 3 * np.pi / 4, -np.pi / 4),
            (3 * np.pi / 4, 0, np.pi / 4),
            (0, np.pi / 2, np.pi / 2),
            (np.pi / 2, 0, -np.pi / 2),
            (0, np.pi, 0),
            (np.pi, 0, 0),
        ]
        for beta, angle, expected in cases:
            with self.subTest(beta=beta, angle=angle):
                self.assertAlmostEqual(gamma(beta, angle), expected)

    def test_invalid_angles(self):
        for angle in [-0.1, np.pi + 0.1, np.nan, np.inf]:
            with self.subTest(angle=angle):
                with self.assertRaises(ValueError):
                    gamma(angle, 0)
                with self.assertRaises(ValueError):
                    gamma(0, angle)

    def test_collinear_tangents_in_all_quadrants(self):
        for posj in [(10, 0), (-10, 0), (0, 10), (0, -10),
                     (10, 10), (-10, 10), (-10, -10), (10, -10)]:
            angle = np.arctan2(posj[1], posj[0]) % np.pi
            with self.subTest(posj=posj):
                self.assertTrue(is_cocircular((0, 0), posj, angle, angle))
                self.assertTrue(is_cocircular(posj, (0, 0), angle, angle))

    def test_tangents_to_same_circle(self):
        # Radius-10 circle centered at (0, 0), sampled at (10, 0) and (0, 10).
        self.assertTrue(is_cocircular((10, 0), (0, 10), np.pi / 2, 0))
        self.assertTrue(is_cocircular((0, 10), (10, 0), 0, np.pi / 2))

    def test_incompatible_tangents(self):
        self.assertFalse(is_cocircular((0, 0), (10, 0), 0, np.pi / 2))
        self.assertFalse(is_cocircular((0, 0), (0, 10), 0, 0))

    def test_adjacent_pixels(self):
        self.assertTrue(is_cocircular((0, 0), (0, 1), 0, np.pi / 2))

    def test_quantization_tolerance(self):
        tolerance = np.pi / 16 + 2 * np.arcsin(1 / 100)
        self.assertTrue(is_cocircular((0, 0), (100, 0), 0, tolerance - 1e-6))
        self.assertFalse(is_cocircular((0, 0), (100, 0), 0, tolerance + 1e-6))

    def test_unsigned_pixel_coordinates(self):
        posi = np.array([10, 10], dtype=np.uint8)
        posj = np.array([0, 0], dtype=np.uint8)
        self.assertTrue(is_cocircular(posi, posj, np.pi / 4, np.pi / 4))

    def test_invalid_positions(self):
        for posj in [(0, 0), (0.5, 0), (np.nan, 0), (np.inf, 0)]:
            with self.subTest(posj=posj), self.assertRaises(ValueError):
                is_cocircular((0, 0), posj, 0, 0)

    def test_coefficient(self):
        self.assertEqual(cocircularity_coeff((0, 0), (0, 10), 8, 8), 1)
        self.assertEqual(cocircularity_coeff((0, 0), (10, 0), 0, 8), 0.1)
        self.assertEqual(cocircularity_coeff((0, 0), (10, 0), 0, 8, c_min=0.2), 0.2)


if __name__ == '__main__':
    unittest.main()
