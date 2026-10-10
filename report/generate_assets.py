"""Regenerate report figures and measurements from the submission code."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from convolution import (
    DEFAULT_KERNEL_PARAMETERS,
    FINAL_VECTOR_COLOR,
    INITIAL_VECTOR_COLOR,
    ORIENTATION_COUNT,
    X_RANGE,
    Y_RANGE,
    convolve_image_with_all_orientations,
    generate_kernel,
    softmax_results,
    tangent_angles,
    tangent_segments,
)
from relaxation_labeling import relax_labels
from support_generation import default_curvature_edges, generate_support

ASSETS = Path(__file__).parent / "assets"
PARAMS = {**DEFAULT_KERNEL_PARAMETERS, "samples": 31}
TEMPERATURE = 0.005
RADIUS = 5
EDGES = default_curvature_edges(RADIUS)
ANGLES = tangent_angles(np.arange(ORIENTATION_COUNT) * np.pi / ORIENTATION_COUNT)
INITIAL_THRESHOLD = 0.2
INITIAL_COMPARISON_THRESHOLD = 0.5
FINAL_THRESHOLD = 0.5
SPACING = 4
IMAGE_ORDER = ("spaghetti", "fingerprint", "hair")


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def overlay(ax, original, probabilities, threshold, title, color, spacing=SPACING):
    ax.imshow(original)
    segments = tangent_segments(ANGLES, probabilities, threshold, spacing)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.6))
    ax.set_title(title)
    ax.axis("off")
    return len(segments)


def kernel_figures():
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), layout="constrained")
    kernels = [
        generate_kernel(n * np.pi / ORIENTATION_COUNT, **PARAMS)
        for n in range(ORIENTATION_COUNT)
    ]
    limit = np.max(np.abs(kernels))
    for n, (ax, kernel) in enumerate(zip(axes.flat, kernels)):
        ax.imshow(kernel, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(
            f"{n * 180 / ORIENTATION_COUNT:g}°", fontsize=20
        )  # Legible when shrunk.
        ax.axis("off")
    save(fig, "kernels_all")

    xs, ys = np.meshgrid(
        np.linspace(*X_RANGE, PARAMS["samples"]),
        np.linspace(*Y_RANGE, PARAMS["samples"]),
    )
    fig = plt.figure(figsize=(9, 3.6), layout="constrained")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(xs, ys, kernels[0], cmap="RdBu_r", linewidth=0)
    # The profile panel labels the value axis, so the 3D z label is omitted
    # to keep it from colliding with the tick labels.
    ax.set(xlabel="x", ylabel="y", title="Reference kernel, 0 degrees")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(xs[0], kernels[0][PARAMS["samples"] // 2], color=INITIAL_VECTOR_COLOR)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set(xlabel="x at y = 0", ylabel="Normalized G", title="Transverse profile")
    ax.grid(alpha=0.2)
    save(fig, "kernel_surface")


def image_figures(name, iterations):
    print(f"{name}: convolution", flush=True)
    path = ROOT / "src/curves" / f"{name}.png"
    with Image.open(path) as source:
        original = np.asarray(source.convert("RGB"))
        image = np.asarray(source.convert("L"), dtype=float) / 255
        source.convert("RGB").save(ASSETS / f"{name}_original.png")
    responses = convolve_image_with_all_orientations(image, PARAMS)
    initial = softmax_results(responses, TEMPERATURE)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), layout="constrained")
    low, high = np.percentile(
        np.stack([r.response[::4, ::4] for r in responses]), [1, 99]
    )
    for ax, result in zip(axes.flat, responses):
        ax.imshow(result.response, cmap="gray", vmin=low, vmax=high)
        ax.set_title(f"{result.index * 180 / ORIENTATION_COUNT:g} degrees", fontsize=9)
        ax.axis("off")
    save(fig, f"{name}_responses")
    del responses

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    initial_count = overlay(
        ax,
        original,
        initial,
        INITIAL_THRESHOLD,
        f"{name.capitalize()}: initial p > 0.20",
        INITIAL_VECTOR_COLOR,
    )
    save(fig, f"{name}_initial")

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    initial_comparison_count = overlay(
        ax,
        original,
        initial,
        INITIAL_COMPARISON_THRESHOLD,
        f"{name.capitalize()}: initial p > 0.50",
        INITIAL_VECTOR_COLOR,
    )
    save(fig, f"{name}_initial_p05")

    print(f"{name}: support", flush=True)
    support, _ = generate_support(initial, ANGLES, RADIUS, EDGES)
    # This normalization is only for the display, never for relaxation.
    display_support = support / max(float(support.max()), 1e-12)
    support_threshold = round(
        float(np.clip(np.percentile(display_support.max(axis=0), 90), 0.01, 0.99)), 2
    )
    supported = np.where(initial > INITIAL_THRESHOLD, display_support, 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    supported_count = overlay(
        axes[0],
        original,
        supported,
        support_threshold,
        f"Supported candidates: s / max > {support_threshold:.2f}",
        FINAL_VECTOR_COLOR,
    )
    heatmap = axes[1].imshow(support.max(axis=0), cmap="magma", vmin=0)
    axes[1].set_title("Maximum raw support over orientations")
    axes[1].axis("off")
    fig.colorbar(heatmap, ax=axes[1], shrink=0.8)
    save(fig, f"{name}_support")

    print(f"{name}: {iterations} relaxation updates", flush=True)
    result = relax_labels(initial, ANGLES, RADIUS, EDGES, iterations=iterations)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    overlay(
        axes[0],
        original,
        initial,
        INITIAL_THRESHOLD,
        "Initial: p > 0.20",
        INITIAL_VECTOR_COLOR,
    )
    final_count = overlay(
        axes[1],
        original,
        result.probabilities,
        FINAL_THRESHOLD,
        f"After {len(result.max_changes)} updates: p > 0.50",
        FINAL_VECTOR_COLOR,
    )
    save(fig, f"{name}_relaxation")

    # Central crops retain image coordinates, so tangent lengths are unchanged.
    height, width = image.shape
    crop = (width // 2 - 64, width // 2 + 64, height // 2 - 64, height // 2 + 64)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    for ax, p, threshold, title, color in (
        (axes[0], initial, INITIAL_THRESHOLD, "Initial", INITIAL_VECTOR_COLOR),
        (
            axes[1],
            supported,
            support_threshold,
            "Supported candidates",
            FINAL_VECTOR_COLOR,
        ),
        (axes[2], result.probabilities, FINAL_THRESHOLD, "Relaxed", FINAL_VECTOR_COLOR),
    ):
        overlay(ax, original, p, threshold, title, color, spacing=2)
        ax.set_xlim(crop[:2])
        ax.set_ylim(crop[3], crop[2])
    save(fig, f"{name}_detail")

    metrics = {
        "image": name,
        "width": width,
        "height": height,
        "initial_vectors": initial_count,
        "supported_vectors": supported_count,
        "initial_vectors_p05": initial_comparison_count,
        "relaxed_vectors": final_count,
        "support_display_threshold": support_threshold,
        "iterations_completed": len(result.max_changes),
        "support_min": result.support_min,
        "support_max": result.support_max,
        "average_local_support": result.average_support,
        "average_local_support_per_pixel": (
            np.asarray(result.average_support) / image.size
        ).tolist(),
        "max_confidence_changes": result.max_changes,
        "central_crop_xyxy": [crop[0], crop[2], crop[1], crop[3]],
    }
    (ASSETS / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{name}: saved figures and measurements", flush=True)
    return metrics


def synthetic_crossing():
    initial = np.zeros((ORIENTATION_COUNT, 31, 31))
    initial[0, 5:26, 15] = 0.7
    initial[8, 15, 5:26] = 0.7
    initial[0, 2, 2] = 0.8
    result = relax_labels(
        initial,
        ANGLES,
        3,
        np.array([-0.05, 0.05]),
        iterations=5,
        c_min=0,
        support_min=1,
        support_max=6,
    )
    original = np.full((31, 31, 3), 255, dtype=np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
    overlay(
        axes[0],
        original,
        initial,
        0.5,
        "Synthetic initial",
        INITIAL_VECTOR_COLOR,
        spacing=1,
    )
    overlay(
        axes[1],
        original,
        result.probabilities,
        0.5,
        "Synthetic relaxed",
        FINAL_VECTOR_COLOR,
        spacing=1,
    )
    axes[2].bar(
        ["Vertical\nat crossing", "Horizontal\nat crossing", "Isolated"],
        [
            result.probabilities[0, 15, 15],
            result.probabilities[8, 15, 15],
            result.probabilities[0, 2, 2],
        ],
        color=[INITIAL_VECTOR_COLOR, FINAL_VECTOR_COLOR, "#71717a"],
    )
    axes[2].set(
        ylabel="Final confidence", ylim=(0, 1), title="Independent orientation labels"
    )
    save(fig, "synthetic_crossing")
    (ASSETS / "synthetic_crossing_metrics.json").write_text(
        json.dumps(
            {
                "radius": 3,
                "curvature_edges": [-0.05, 0.05],
                "c_min": 0,
                "support_min": 1,
                "support_max": 6,
                "iterations": 5,
                "crossing_confidences": result.probabilities[[0, 8], 15, 15].tolist(),
                "isolated_confidence": float(result.probabilities[0, 2, 2]),
            },
            indent=2,
        )
        + "\n"
    )


def summary_figures(metrics):
    fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    with (ASSETS / "support_history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["image", "iteration", "A_p", "A_p_per_pixel", "max_confidence_change"]
        )
        for m in metrics:
            values = m["average_local_support_per_pixel"]
            ax.plot(range(len(values)), values, marker="o", label=m["image"])
            for i, (raw, average) in enumerate(zip(m["average_local_support"], values)):
                writer.writerow(
                    [
                        m["image"],
                        i,
                        raw,
                        average,
                        m["max_confidence_changes"][i - 1] if i else "",
                    ]
                )
    ax.set(
        xlabel="Iteration",
        ylabel="A(p) / number of pixels",
        title="Average local support",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, "support_history")

    rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Image & Initial & Supported & Relaxed & $A_0/N$ & $A_T/N$ \\",
        r"\midrule",
    ]
    for m in metrics:
        values = m["average_local_support_per_pixel"]
        rows.append(
            f"{m['image'].capitalize()} & {m['initial_vectors']:,} & "
            f"{m['supported_vectors']:,} & {m['relaxed_vectors']:,} & "
            f"{values[0]:.3f} & {values[-1]:.3f} " + r"\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    (ASSETS / "results_table.tex").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    kernel_figures()
    synthetic_crossing()
    metrics = [image_figures(name, args.iterations) for name in IMAGE_ORDER]
    summary_figures(metrics)
    manifest = {
        "image_order": list(IMAGE_ORDER),
        "kernel_parameters": PARAMS,
        "temperature": TEMPERATURE,
        "orientation_rotations_radians": (
            np.arange(ORIENTATION_COUNT) * np.pi / ORIENTATION_COUNT
        ).tolist(),
        "tangent_angles_radians": ANGLES.tolist(),
        "neighborhood_radius": RADIUS,
        "curvature_edges": EDGES.tolist(),
        "c_min": 0.1,
        "iterations_requested": args.iterations,
        "step_size": 1,
        "tolerance": 1e-4,
        "initial_threshold": INITIAL_THRESHOLD,
        "final_threshold": FINAL_THRESHOLD,
        "initial_comparison_threshold": INITIAL_COMPARISON_THRESHOLD,
        "vector_spacing": SPACING,
        "detail_vector_spacing": 2,
        "support_display_threshold": (
            "90th percentile of normalized per-pixel maxima, rounded to 0.01"
        ),
        "support_candidate_filter": "initial confidence > 0.20",
        "input_processing": (
            "Pillow grayscale divided by 255; original resolution; symmetric padding"
        ),
        "label_model": (
            "independent tangent versus no-line at each orientation after initialization"
        ),
        "measurements": metrics,
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
