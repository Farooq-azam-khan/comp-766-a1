import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from convolution import (
    DEFAULT_KERNEL_PARAMETERS,
    FINAL_VECTOR_COLOR,
    INITIAL_VECTOR_COLOR,
    ORIENTATION_COUNT,
    convolve_image_with_all_orientations,
    softmax_results,
    tangent_angles,
    tangent_segments,
)
from support_generation import (
    accumulate_support,
    consistent_support,
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
    # Eq. 6.15 with minimum line confidence 0.6: s_min = 0.6 * s_max / 2.
    support_min = float(0.3 * support_max if support_min is None else support_min)
    offsets = support_offsets(angles, radius, curvature_edges, c_min)
    class_count = len(curvature_edges) - 1
    support, classes = consistent_support(probabilities, offsets, class_count)
    scores = [average_local_support(probabilities, support)]
    changes = []
    for _ in range(iterations):
        signed_support = (support - support_min) / (support_max - support_min)
        updated = radial_update(probabilities, signed_support, step_size)
        changes.append(float(np.max(np.abs(updated - probabilities))))
        probabilities = updated
        support, classes = accumulate_support(
            probabilities, offsets, class_count, classes
        )
        scores.append(average_local_support(probabilities, support))
        if changes[-1] <= tolerance:
            break
    return RelaxationResult(
        probabilities, support, classes, scores, changes, support_min, support_max
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run relaxation labeling and save arrays and a preview."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "curves/fingerprint.png",
    )
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--max-curvature", type=float, default=0.2)
    parser.add_argument("--c-min", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.005)
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
    angles = tangent_angles(
        np.arange(ORIENTATION_COUNT, dtype=np.float64) * np.pi / ORIENTATION_COUNT
    )
    edges = default_curvature_edges(args.radius, args.max_curvature, args.classes)
    params = DEFAULT_KERNEL_PARAMETERS.copy()
    params["samples"] = args.samples
    with Image.open(args.image) as source:
        original = np.asarray(source.convert("RGB"), dtype=np.uint8)
        image = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
    if args.assignments is None:
        convolutions = convolve_image_with_all_orientations(image, params)
        initial = softmax_results(convolutions, temperature=args.temperature)
        del convolutions
    else:
        initial = np.load(args.assignments, allow_pickle=False).astype(
            np.float64, copy=False
        )
    print("Running relaxation labeling...", flush=True)
    result = relax_labels(
        initial,
        angles,
        args.radius,
        edges,
        iterations=args.iterations,
        step_size=args.step_size,
        tolerance=args.tolerance,
        support_min=args.support_min,
        support_max=args.support_max,
        c_min=args.c_min,
    )

    import matplotlib

    if args.save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"Relaxation labeling: {args.image.name}")
    for ax, probabilities, threshold, title, color in (
        (
            axes[0],
            initial,
            args.initial_threshold,
            "Initial tangents",
            INITIAL_VECTOR_COLOR,
        ),
        (
            axes[1],
            result.probabilities,
            args.threshold,
            "Relaxed tangents",
            FINAL_VECTOR_COLOR,
        ),
    ):
        ax.imshow(original)
        segments = tangent_segments(angles, probabilities, threshold, args.spacing)
        ax.add_collection(LineCollection(list(segments), colors=color, linewidths=0.8))
        ax.set_title(f"{title}, p > {threshold:.2f}\n{len(segments):,} vectors")
        ax.axis("off")
    per_pixel_scores = np.asarray(result.average_support) / image.size
    axes[2].plot(np.arange(len(per_pixel_scores)), per_pixel_scores, marker="o")
    axes[2].set(
        xlabel="Iteration",
        ylabel="A(p) / number of pixels",
        title="Average local support",
    )
    axes[2].grid(alpha=0.25)
    fig.tight_layout()

    output = args.output or Path("output/relaxation") / args.image.stem
    metadata = {
        "source": str(args.image),
        "assignments": str(args.assignments) if args.assignments else None,
        "kernel_parameters": params if args.assignments is None else None,
        "temperature": args.temperature if args.assignments is None else None,
        "neighborhood_radius": args.radius,
        "curvature_edges": edges.tolist(),
        "c_min": args.c_min,
        "iterations_requested": args.iterations,
        "iterations_completed": len(result.max_changes),
        "step_size": args.step_size,
        "tolerance": args.tolerance,
        "support_min": result.support_min,
        "support_max": result.support_max,
        "average_local_support": result.average_support,
        "average_local_support_per_pixel": per_pixel_scores.tolist(),
        "max_confidence_changes": result.max_changes,
        "initial_threshold": args.initial_threshold,
        "threshold": args.threshold,
        "vector_spacing": args.spacing,
        "tangent_angles_radians": angles.tolist(),
        "array_axes": ["orientation", "row", "column"],
        "label_model": "independent tangent versus no-line at each orientation",
    }
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "initial_assignments.npy", initial)
    np.save(output / "assignments.npy", result.probabilities)
    np.save(output / "support.npy", result.support)
    np.save(output / "curvature_classes.npy", result.curvature_classes)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig.savefig(output / "preview.png", dpi=150)
    for iteration, score in enumerate(result.average_support):
        print(f"Iteration {iteration}: A(p) = {score:.6g}")
    print(f"Saved to {output}", flush=True)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
