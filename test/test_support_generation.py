import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from numpy.typing import NDArray

from support_generation import (
    cocircularity_coeff, default_curvature_edges, gamma, generate_support, is_cocircular,
)


def direct_support(
    probabilities: NDArray[np.float64],
    angles: NDArray[np.float64],
    radius: int,
    edges: NDArray[np.float64],
    previous_classes: NDArray[np.int64] | None,
    c_min: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    label_count, height, width = probabilities.shape
    class_count = len(edges) - 1
    accumulated = np.zeros((class_count, label_count, height, width))

    def class_for(angle: float, dx: int, dy: int) -> int:
        curvature = 2 * (np.cos(angle) * dy - np.sin(angle) * dx) / (dx * dx + dy * dy)
        for index in range(class_count):
            if edges[index] <= curvature < edges[index + 1]:
                return index
        return class_count - 1 if curvature == edges[-1] else -1

    for y in range(height):
        for x in range(width):
            for label, angle in enumerate(angles):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if not 0 < dx * dx + dy * dy <= radius * radius:
                            continue
                        neighbor_x, neighbor_y = x + dx, y + dy
                        if not (0 <= neighbor_x < width and 0 <= neighbor_y < height):
                            continue
                        target_class = class_for(float(angle), dx, dy)
                        if target_class < 0:
                            continue
                        for neighbor_label, neighbor_angle in enumerate(angles):
                            if previous_classes is not None:
                                neighbor_class = class_for(float(neighbor_angle), -dx, -dy)
                                if neighbor_class < 0 or previous_classes[
                                    neighbor_label, neighbor_y, neighbor_x
                                ] != neighbor_class:
                                    continue
                            coefficient = 1.0 if is_cocircular(
                                (x, y), (neighbor_x, neighbor_y),
                                float(angle), float(neighbor_angle),
                            ) else c_min
                            accumulated[target_class, label, y, x] += (
                                coefficient * probabilities[neighbor_label, neighbor_y, neighbor_x]
                            )
    support = accumulated.max(axis=0)
    return support, np.where(support > 0, accumulated.argmax(axis=0), -1)


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

    def test_coefficient(self):
        self.assertEqual(cocircularity_coeff((0, 0), (0, 10), 8, 8), 1)
        self.assertEqual(cocircularity_coeff((0, 0), (10, 0), 0, 8), 0.1)
        self.assertEqual(cocircularity_coeff((0, 0), (10, 0), 0, 8, c_min=0.2), 0.2)


class SupportTests(unittest.TestCase):
    def test_default_bins_preserve_straight_tangents_at_all_orientations(self) -> None:
        from convolution import tangent_angles

        angles = tangent_angles(np.arange(16) * np.pi / 16)
        y, x = np.mgrid[-16:17, -16:17]
        edges = default_curvature_edges(5)
        for label, angle in enumerate(angles):
            normal_distance = -np.sin(angle) * x + np.cos(angle) * y
            strength = 0.5 * np.exp(-normal_distance ** 2 / 9)
            probabilities = np.full((16, 33, 33), 1 / 16)
            probabilities *= 1 - strength
            probabilities[label] += strength
            with self.subTest(label=label):
                support, classes = generate_support(probabilities, angles, 5, edges)
                self.assertEqual(int(support[:, 16, 16].argmax()), label)
                self.assertEqual(classes[label, 16, 16], 3)

    def test_default_bin_configuration(self) -> None:
        edges = default_curvature_edges(5)
        self.assertEqual(len(edges), 8)
        self.assertAlmostEqual(edges[3], -0.08)
        self.assertAlmostEqual(edges[4], 0.08)
        np.testing.assert_array_equal(default_curvature_edges(5, class_count=1), [-0.2, 0.2])

    def test_matches_direct_equation_with_previous_classes(self) -> None:
        rng = np.random.default_rng(7)
        probabilities = rng.random((4, 3, 5))
        angles = np.arange(4) * np.pi / 4
        edges = np.array([-2.0, -0.2, 0.2, 2.0])
        previous = rng.integers(-1, 3, size=probabilities.shape)
        expected, expected_classes = direct_support(probabilities, angles, 2, edges, previous, 0.1)
        actual, actual_classes = generate_support(
            probabilities, angles, 2, edges, curvature_classes=previous
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_array_equal(actual_classes, expected_classes)

    def test_bootstrap_matches_two_direct_passes(self) -> None:
        probabilities = np.random.default_rng(12).random((4, 4, 3))
        angles = np.arange(4) * np.pi / 4
        edges = np.array([-2.0, -0.25, 0.25, 2.0])
        _, initial_classes = direct_support(probabilities, angles, 2, edges, None, 0.1)
        expected, expected_classes = direct_support(probabilities, angles, 2, edges, initial_classes, 0.1)
        actual, actual_classes = generate_support(probabilities, angles, 2, edges)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_array_equal(actual_classes, expected_classes)

    def test_straight_line_supports_its_tangent(self) -> None:
        probabilities = np.zeros((2, 7, 9))
        probabilities[0, 3, :] = 1.0
        support, classes = generate_support(
            probabilities, np.array([0.0, np.pi / 2]), 3,
            np.array([-0.5, -0.1, 0.1, 0.5]), c_min=0.0,
        )
        self.assertAlmostEqual(support[0, 3, 4], 6.0)
        self.assertEqual(classes[0, 3, 4], 1)
        self.assertEqual(support[1, 3, 4], 0.0)

    def test_no_self_support_or_wrapped_edges(self) -> None:
        probabilities = np.zeros((1, 1, 5))
        probabilities[0, 0, 0] = 1.0
        support, classes = generate_support(
            probabilities, np.array([0.0]), 1, np.array([-3.0, 3.0])
        )
        self.assertEqual(support[0, 0, 0], 0.0)
        self.assertEqual(support[0, 0, -1], 0.0)
        self.assertEqual(classes[0, 0, -1], -1)

    def test_zero_assignments(self) -> None:
        probabilities = np.zeros((2, 3, 4))
        support, classes = generate_support(
            probabilities, np.array([0.0, np.pi / 2]), 2, np.array([-1.0, 1.0])
        )
        np.testing.assert_array_equal(support, 0.0)
        np.testing.assert_array_equal(classes, -1)

    def test_inconsistent_neighbor_classes_remove_support(self) -> None:
        probabilities = np.ones((1, 1, 5))
        support, _ = generate_support(
            probabilities, np.array([0.0]), 2, np.array([-0.1, 0.1]),
            curvature_classes=np.full(probabilities.shape, -1, dtype=np.int64),
        )
        np.testing.assert_array_equal(support, 0.0)


class SupportVisualizationTests(unittest.TestCase):
    def test_support_overlay_filters_initial_candidates(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from support_ui import SupportUI

        probabilities = np.full((2, 3, 3), 0.1)
        probabilities[0, 1, 1] = 0.9
        support = np.ones_like(probabilities)
        with TemporaryDirectory() as directory:
            ui = SupportUI(
                np.zeros((3, 3, 3), dtype=np.uint8), probabilities,
                np.array([0.0, np.pi / 2]), support,
                np.zeros(probabilities.shape, dtype=np.int64),
                Path(directory), {"source": "synthetic"},
            )
            try:
                ui.spacing.set_val(1)
                ui.support_threshold.set_val(0.5)
                self.assertEqual(len(ui.initial_artist.get_segments()), 1)
                self.assertEqual(len(ui.support_artist.get_segments()), 1)
                ui.candidate_filter.set_active(0)
                self.assertEqual(len(ui.support_artist.get_segments()), 18)
                np.testing.assert_array_equal(ui.support, support)
            finally:
                plt.close(ui.fig)

    def test_controls_and_export_preserve_raw_support(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from support_ui import SupportUI

        angles = np.array([0.0, np.pi / 2])
        probabilities = np.zeros((2, 7, 9))
        probabilities[0, 3, :] = 1.0
        support, classes = generate_support(
            probabilities, angles, 3, np.array([-0.5, -0.1, 0.1, 0.5]), c_min=0.0,
        )
        expected_support = support.copy()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            ui = SupportUI(
                np.full((7, 9, 3), 255, dtype=np.uint8),
                probabilities, angles, support, classes, output, {"source": "synthetic"},
            )
            try:
                ui.spacing.set_val(1)
                ui.support_threshold.set_val(0.5)
                self.assertGreater(len(ui.support_artist.get_segments()), 0)
                ui.support_threshold.set_val(1.0)
                self.assertEqual(len(ui.support_artist.get_segments()), 0)
                ui.support_threshold.set_val(0.5)
                self.assertTrue(ui.save())
                np.testing.assert_array_equal(np.load(output / "support.npy"), expected_support)
                np.testing.assert_array_equal(np.load(output / "curvature_classes.npy"), classes)
                self.assertGreater((output / "preview.png").stat().st_size, 0)
                metadata = json.loads((output / "manifest.json").read_text())
                self.assertEqual(metadata["support_threshold"], 0.5)
                np.testing.assert_array_equal(support, expected_support)
            finally:
                plt.close(ui.fig)


if __name__ == '__main__':
    unittest.main()
