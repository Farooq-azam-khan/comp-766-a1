import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from convolution import (
    DEFAULT_KERNEL_PARAMETERS,
    ORIENTATION_COUNT,
    convolve_image_with_all_orientations,
    softmax_results,
    tangent_angles,
    tangent_segments,
)
from support_generation import (
    accumulate_support,
    default_curvature_edges,
    support_offsets,
)


@dataclass(frozen=True)
class RelaxationResult:
    probabilities: NDArray[np.float64]
    support: NDArray[np.float64]
    curvature_classes: NDArray[np.int64]
    average_support: list[float]
    max_changes: list[float]
    support_min: float
    support_max: float


def average_local_support(
    probabilities: NDArray[np.float64], support: NDArray[np.float64]
) -> float:
    """Return A(p) = sum over pixels and orientations of p * s."""
    return float(np.sum(probabilities * support))


def radial_update(
    probabilities: NDArray[np.float64],
    signed_support: NDArray[np.float64],
    step_size: float = 1.0,
) -> NDArray[np.float64]:
    # Each orientation competes with its own no-line label, with support -s.
    # Shifting [s, -s] by its minimum gives [2*max(s, 0), 2*max(-s, 0)].
    return (probabilities + 2 * step_size * np.maximum(signed_support, 0)) / (
        1 + 2 * step_size * np.abs(signed_support)
    )


def relax_labels(
    probabilities: NDArray[np.float64],
    angles: NDArray[np.float64],
    radius: int,
    curvature_edges: NDArray[np.float64],
    *,
    iterations: int = 5,
    step_size: float = 1.0,
    tolerance: float = 1e-4,
    support_min: float | None = None,
    support_max: float | None = None,
    c_min: float = 0.1,
) -> RelaxationResult:
    probabilities = np.asarray(probabilities, dtype=np.float64).copy()
    # A unit-confidence straight line through a radius-r neighborhood has 2r neighbors.
    support_max = float(2 * radius if support_max is None else support_max)
    # Eq. 6.15 with minimum line confidence 0.5: s_min = 0.5 * s_max / 2.
    support_min = float(0.25 * support_max if support_min is None else support_min)
    offsets = support_offsets(angles, radius, curvature_edges, c_min)
    class_count = len(curvature_edges) - 1
    _, classes = accumulate_support(probabilities, offsets, class_count, None)
    support, classes = accumulate_support(probabilities, offsets, class_count, classes)
    scores = [average_local_support(probabilities, support)]
    changes = []
    for _ in range(iterations):
        signed_support = (support - support_min) / (support_max - support_min)
        updated = radial_update(probabilities, signed_support, step_size)
        changes.append(float(np.max(np.abs(updated - probabilities))))
        probabilities = updated
        support, classes = accumulate_support(probabilities, offsets, class_count, classes)
        scores.append(average_local_support(probabilities, support))
        if changes[-1] <= tolerance:
            break
    return RelaxationResult(probabilities, support, classes, scores, changes, support_min, support_max)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=Path("curves/fingerprint.png"))
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--max-curvature", type=float, default=0.2)
    parser.add_argument("--c-min", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--support-min", type=float)
    parser.add_argument("--support-max", type=float)
    parser.add_argument("--initial-threshold", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--spacing", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Temperature must be finite and positive.")
    if args.radius < 1 or args.classes < 1 or args.classes % 2 == 0:
        parser.error("Radius must be positive and classes must be a positive odd number.")
    if not np.isfinite(args.max_curvature) or args.max_curvature <= 0:
        parser.error("Maximum curvature must be finite and positive.")
    if args.spacing < 1 or not 0 <= args.threshold <= 1 or not 0 <= args.initial_threshold <= 1:
        parser.error("Spacing must be positive and display thresholds between 0 and 1.")
    angles = tangent_angles(np.arange(ORIENTATION_COUNT, dtype=np.float64) * np.pi / ORIENTATION_COUNT)
    edges = default_curvature_edges(args.radius, args.max_curvature, args.classes)
    try:
        with Image.open(args.image) as source:
            original = np.asarray(source.convert("RGB"), dtype=np.uint8)
            image = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
        if args.assignments is None:
            convolutions = convolve_image_with_all_orientations(image, DEFAULT_KERNEL_PARAMETERS)
            initial = softmax_results(convolutions, temperature=args.temperature)
            del convolutions
        else:
            initial = np.load(args.assignments, allow_pickle=False).astype(np.float64, copy=False)
        if initial.shape != (ORIENTATION_COUNT, *image.shape):
            raise ValueError("Assignments must have shape (16, image height, image width).")
        print("Running relaxation labeling...", flush=True)
        result = relax_labels(
            initial, angles, args.radius, edges, iterations=args.iterations,
            step_size=args.step_size, tolerance=args.tolerance,
            support_min=args.support_min, support_max=args.support_max, c_min=args.c_min,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    import matplotlib

    if args.save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"Relaxation labeling: {args.image.name}")
    for ax, probabilities, threshold, title, color in (
        (axes[0], initial, args.initial_threshold, "Initial tangents", "#2563eb"),
        (axes[1], result.probabilities, args.threshold, "Relaxed tangents", "#e11d48"),
    ):
        ax.imshow(original)
        segments = tangent_segments(angles, probabilities, threshold, args.spacing)
        ax.add_collection(LineCollection(list(segments), colors=color, linewidths=0.8))
        ax.set_title(f"{title}, p > {threshold:.2f}\n{len(segments):,} vectors")
        ax.axis("off")
    per_pixel_scores = np.asarray(result.average_support) / image.size
    axes[2].plot(np.arange(len(per_pixel_scores)), per_pixel_scores, marker="o")
    axes[2].set(xlabel="Iteration", ylabel="A(p) / number of pixels", title="Average local support")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()

    output = args.output or Path("output/relaxation") / args.image.stem
    metadata = {
        "source": str(args.image),
        "assignments": str(args.assignments) if args.assignments else None,
        "kernel_parameters": DEFAULT_KERNEL_PARAMETERS if args.assignments is None else None,
        "temperature": args.temperature if args.assignments is None else None,
        "neighborhood_radius": args.radius,
        "curvature_edges": edges.tolist(), "c_min": args.c_min,
        "iterations_requested": args.iterations, "iterations_completed": len(result.max_changes),
        "step_size": args.step_size, "tolerance": args.tolerance,
        "support_min": result.support_min, "support_max": result.support_max,
        "average_local_support": result.average_support,
        "average_local_support_per_pixel": per_pixel_scores.tolist(),
        "max_confidence_changes": result.max_changes,
        "initial_threshold": args.initial_threshold, "threshold": args.threshold,
        "vector_spacing": args.spacing, "tangent_angles_radians": angles.tolist(),
        "array_axes": ["orientation", "row", "column"],
        "label_model": "independent tangent versus no-line at each orientation",
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "initial_assignments.npy", initial)
        np.save(output / "assignments.npy", result.probabilities)
        np.save(output / "support.npy", result.support)
        np.save(output / "curvature_classes.npy", result.curvature_classes)
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        fig.savefig(output / "preview.png", dpi=150)
    except OSError as exc:
        parser.exit(1, f"Save failed: {exc}\n")
    for iteration, score in enumerate(result.average_support):
        print(f"Iteration {iteration}: A(p) = {score:.6g}")
    print(f"Saved to {output}", flush=True)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
