import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from relaxation_labeling import average_local_support, radial_update, relax_labels
from support_generation import default_curvature_edges, generate_support


class RelaxationTests(unittest.TestCase):
    def test_average_local_support_sums_pixels_and_orientations(self):
        probabilities = np.array([[[0.8, 0.3]], [[0.2, 0.7]]])
        support = np.array([[[0.9, 0.4]], [[0.1, 0.6]]])
        self.assertAlmostEqual(average_local_support(probabilities, support), 1.28)

    def test_binary_rule_matches_appendix_vector_projection(self):
        probabilities = np.array([0.3, 0.3, 0.3, 0.0, 1.0])
        support = np.array([0.4, -0.4, 0.0, 0.4, -0.4])
        for step in (0.25, 1.0, 4.0):
            labels = np.stack((probabilities, 1 - probabilities))
            evidence = np.stack((support, -support))
            evidence -= np.minimum(evidence.min(axis=0), 0)
            expected = (labels + step * evidence) / (1 + step * evidence.sum(axis=0))
            actual = radial_update(probabilities, support, step)
            np.testing.assert_allclose(actual, expected[0])
            self.assertTrue(np.all((actual >= 0) & (actual <= 1)))
        self.assertAlmostEqual(radial_update(probabilities, support)[0], 0.611111111111)
        self.assertAlmostEqual(radial_update(probabilities, support)[1], 0.166666666667)

    def test_crossing_survives_and_isolated_tangent_decays(self):
        probabilities = np.zeros((2, 21, 21))
        probabilities[0, 10, 3:18] = 0.6
        probabilities[1, 3:18, 10] = 0.6
        probabilities[0, 1, 1] = 0.8
        initial = probabilities.copy()
        result = relax_labels(
            probabilities,
            np.array([0.0, np.pi / 2]),
            3,
            np.array([-0.05, 0.05]),
            iterations=5,
            c_min=0.0,
            support_min=1.0,
            support_max=6.0,
        )
        self.assertTrue(np.all(result.probabilities[:, 10, 10] > 0.9))
        self.assertLess(result.probabilities[0, 1, 1], 0.2)
        self.assertGreater(result.probabilities[:, 10, 10].sum(), 1.0)
        np.testing.assert_array_equal(probabilities, initial)

    def test_iterations_recompute_support_and_carry_curvature_classes(self):
        initial = np.random.default_rng(4).random((4, 5, 6))
        angles = np.arange(4) * np.pi / 4
        edges = np.array([-2.0, -0.2, 0.2, 2.0])
        result = relax_labels(
            initial,
            angles,
            2,
            edges,
            iterations=3,
            tolerance=0.0,
            step_size=0.5,
            support_min=1.0,
            support_max=4.0,
        )
        probabilities = initial.copy()
        support, classes = generate_support(probabilities, angles, 2, edges)
        scores = [average_local_support(probabilities, support)]
        changes = []
        for _ in range(3):
            updated = radial_update(probabilities, (support - 1) / 3, 0.5)
            changes.append(float(np.max(np.abs(updated - probabilities))))
            probabilities = updated
            support, classes = generate_support(
                probabilities,
                angles,
                2,
                edges,
                curvature_classes=classes,
            )
            scores.append(average_local_support(probabilities, support))
        np.testing.assert_allclose(result.probabilities, probabilities)
        np.testing.assert_allclose(result.support, support)
        np.testing.assert_array_equal(result.curvature_classes, classes)
        np.testing.assert_allclose(result.average_support, scores)
        np.testing.assert_allclose(result.max_changes, changes)

    def test_zero_iterations_and_convergence(self):
        probabilities = np.zeros((1, 3, 3))
        for iterations, expected_steps in ((0, 0), (5, 1)):
            result = relax_labels(
                probabilities,
                np.array([0.0]),
                1,
                np.array([-1.0, 1.0]),
                iterations=iterations,
            )
            np.testing.assert_array_equal(result.probabilities, probabilities)
            self.assertEqual(len(result.max_changes), expected_steps)
            self.assertEqual(len(result.average_support), expected_steps + 1)

    def test_default_threshold_suppresses_diffuse_initial_confidences(self):
        initial = np.full((16, 15, 15), 1 / 16)
        result = relax_labels(
            initial,
            np.arange(16) * np.pi / 16,
            5,
            default_curvature_edges(5),
            iterations=5,
        )
        self.assertLessEqual(float(result.probabilities.max()), 1 / 16)

    def test_save_only_cli_exports_final_state_and_history(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "crossing.png"
            assignments_path = root / "initial.npy"
            output = root / "output"
            Image.fromarray(np.full((11, 11, 3), 255, dtype=np.uint8)).save(image_path)
            initial = np.zeros((16, 11, 11))
            initial[0, :, 5] = 0.7
            initial[8, 5, :] = 0.7
            np.save(assignments_path, initial)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[1]
                        / "src/relaxation_labeling.py"
                    ),
                    str(image_path),
                    "--assignments",
                    str(assignments_path),
                    "--radius",
                    "2",
                    "--classes",
                    "1",
                    "--iterations",
                    "2",
                    "--output",
                    str(output),
                    "--save-only",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            metadata = json.loads((output / "manifest.json").read_text())
            final = np.load(output / "assignments.npy")
            support = np.load(output / "support.npy")
            self.assertEqual(metadata["iterations_completed"], 2)
            self.assertEqual(len(metadata["average_local_support"]), 3)
            self.assertAlmostEqual(
                metadata["average_local_support"][-1],
                average_local_support(final, support),
            )
            np.testing.assert_array_equal(
                np.load(output / "initial_assignments.npy"), initial
            )
            self.assertEqual(
                np.load(output / "curvature_classes.npy").shape, initial.shape
            )
            self.assertTrue(np.all((final >= 0) & (final <= 1)))
            with Image.open(output / "preview.png") as preview:
                self.assertGreater(preview.width, 1000)
            self.assertIn("Iteration 0: A(p)", completed.stdout)

    def test_default_pipeline_recovers_fine_parallel_ridges(self):
        # Ten-pixel ridge spacing exposes the averaging caused by a 101-pixel kernel.
        _, cols = np.mgrid[:65, :65]
        image = 0.65 - 0.15 * np.cos(2 * np.pi * (cols - 32) / 10)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "ridges.png"
            output = root / "results"
            Image.fromarray(np.round(image * 255).astype(np.uint8)).save(image_path)
            subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[1]
                        / "src/relaxation_labeling.py"
                    ),
                    str(image_path),
                    "--output",
                    str(output),
                    "--save-only",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            initial = np.load(output / "initial_assignments.npy")
            final = np.load(output / "assignments.npy")
            self.assertGreater(initial[0, 32, 32], 0.9)
            self.assertGreater(final[0, 32, 32], initial[0, 32, 32])
            self.assertLess(float(final[1:, 32, 32].max()), 0.2)


if __name__ == "__main__":
    unittest.main()
