import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, CheckButtons, Slider
from numpy.typing import NDArray
from PIL import Image

from convolution import (
    ORIENTATION_COUNT,
    KernelParameters,
    convolve_image_with_all_orientations,
    softmax_results,
    tangent_angles,
    tangent_segments,
)
from support_generation import default_curvature_edges, generate_support


class SupportUI:
    def __init__(
        self,
        original: NDArray[np.uint8],
        probabilities: NDArray[np.float64],
        angles: NDArray[np.float64],
        support: NDArray[np.float64],
        curvature_classes: NDArray[np.int64],
        output: Path,
        metadata: dict[str, object],
    ) -> None:
        self.probabilities = probabilities
        self.angles = angles
        self.support = support
        self.curvature_classes = curvature_classes
        maximum = float(support.max())
        self.display_support = support / maximum if maximum > 0 else support.copy()
        default_threshold = round(float(np.clip(
            np.percentile(self.display_support.max(axis=0), 90), 0.01, 0.99
        )), 2)
        self.output = output
        self.metadata = metadata
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 7))
        self.fig.subplots_adjust(left=0.03, right=0.97, bottom=0.25, top=0.90, wspace=0.15)
        self.fig.suptitle(f"Cocircularity support · {metadata['source']}", fontsize=14)
        for ax in self.axes[:2]:
            ax.imshow(original)
        self.initial_artist = LineCollection([], colors="#2563eb", linewidths=0.9)
        self.support_artist = LineCollection([], colors="#e11d48", linewidths=0.9)
        self.axes[0].add_collection(self.initial_artist)
        self.axes[1].add_collection(self.support_artist)
        heatmap = self.axes[2].imshow(support.max(axis=0), cmap="magma", vmin=0, vmax=max(maximum, 1e-12))
        self.axes[2].set_title("Maximum raw support across orientations")
        self.fig.colorbar(heatmap, ax=self.axes[2], fraction=0.046, pad=0.04)
        for ax in self.axes:
            ax.axis("off")
        self.initial_threshold = Slider(
            self.fig.add_axes((0.13, 0.16, 0.24, 0.025)),
            "Initial threshold", 0.0, 1.0, valinit=0.2, valstep=0.01,
        )
        self.support_threshold = Slider(
            self.fig.add_axes((0.60, 0.16, 0.24, 0.025)),
            "Support threshold", 0.0, 1.0, valinit=default_threshold, valstep=0.01,
        )
        self.spacing = Slider(
            self.fig.add_axes((0.13, 0.08, 0.24, 0.025)),
            "Vector spacing", 1, 16, valinit=4, valstep=1, valfmt="%d px",
        )
        self.candidate_filter = CheckButtons(
            self.fig.add_axes((0.45, 0.065, 0.22, 0.06)),
            ["Initial candidates only"], [True],
        )
        self.candidate_filter.on_clicked(self.refresh)
        self.save_button = Button(self.fig.add_axes((0.70, 0.065, 0.14, 0.05)), "Save results")
        self.status = self.fig.text(0.03, 0.02, "", fontsize=9)
        for slider in (self.initial_threshold, self.support_threshold, self.spacing):
            slider.on_changed(self.refresh)
        self.save_button.on_clicked(self.save)
        self.refresh()

    def refresh(self, _value: float | str | None = None) -> None:
        spacing = int(self.spacing.val)
        initial_threshold = float(self.initial_threshold.val)
        initial_segments = tangent_segments(
            self.angles, self.probabilities, initial_threshold, spacing
        )
        candidates_only = self.candidate_filter.get_status()[0]
        displayed_support = (
            np.where(self.probabilities > initial_threshold, self.display_support, 0.0)
            if candidates_only else self.display_support
        )
        support_segments = tangent_segments(
            self.angles, displayed_support, float(self.support_threshold.val), spacing
        )
        self.initial_artist.set_segments(list(initial_segments))
        self.support_artist.set_segments(list(support_segments))
        self.axes[0].set_title(
            f"Initial tangents · p > {self.initial_threshold.val:.2f}\n{len(initial_segments):,} vectors",
            fontsize=10,
        )
        title = (
            f"Supported initial tangents · p > {initial_threshold:.2f}"
            if candidates_only else "Support hypotheses"
        )
        self.axes[1].set_title(
            f"{title}\ns / global max > {self.support_threshold.val:.2f} · "
            f"{len(support_segments):,} vectors",
            fontsize=10,
        )
        self.fig.canvas.draw_idle()

    def save(self, _event: object = None) -> bool:
        try:
            self.output.mkdir(parents=True, exist_ok=True)
            np.save(self.output / "support.npy", self.support)
            np.save(self.output / "curvature_classes.npy", self.curvature_classes)
            metadata = {
                **self.metadata,
                "tangent_angles_radians": self.angles.tolist(),
                "support_axes": ["orientation", "row", "column"],
                "display_normalization": "raw support divided by its global maximum",
                "initial_threshold": float(self.initial_threshold.val),
                "support_threshold": float(self.support_threshold.val),
                "vector_spacing": int(self.spacing.val),
                "initial_candidates_only": bool(self.candidate_filter.get_status()[0]),
                "unsupported_curvature_class": -1,
            }
            (self.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
            self.status.set_text(f"Saved to {self.output}")
            self.status.set_color("black")
            self.fig.savefig(self.output / "preview.png", dpi=150)
        except OSError as exc:
            self.status.set_text(f"Save failed: {exc}")
            self.status.set_color("firebrick")
            self.fig.canvas.draw_idle()
            return False
        self.fig.canvas.draw_idle()
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and visualize cocircularity support.")
    parser.add_argument("image", nargs="?", type=Path, default=Path("curves/spaghetti.png"))
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--max-curvature", type=float, default=0.2)
    parser.add_argument("--c-min", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Temperature must be finite and positive.")
    params: KernelParameters = {
        "samples": 101, "sigma_y": 1.0, "sigma_1": 0.8, "sigma_2": 0.3,
        "sigma_3": 0.8, "A": 0.5, "B": 1.5, "C": 0.5,
    }
    try:
        with Image.open(args.image) as source:
            original = np.asarray(source.convert("RGB"), dtype=np.uint8)
            image = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
        if args.assignments is None:
            results = convolve_image_with_all_orientations(image, params)
            probabilities = softmax_results(results, temperature=args.temperature)
        else:
            probabilities = np.load(args.assignments, allow_pickle=False).astype(np.float64, copy=False)
        if probabilities.shape != (ORIENTATION_COUNT, *image.shape):
            raise ValueError('Assignments must have shape (16, image height, image width).')
        angles = tangent_angles(np.arange(ORIENTATION_COUNT, dtype=np.float64) * np.pi / ORIENTATION_COUNT)
        edges = default_curvature_edges(args.radius, args.max_curvature, args.classes)
        print("Computing cocircularity support...", flush=True)
        support, classes = generate_support(
            probabilities, angles, args.radius, edges, c_min=args.c_min
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output or Path("output/support") / args.image.stem
    metadata: dict[str, object] = {
        "source": str(args.image),
        "assignments": str(args.assignments) if args.assignments else None,
        "kernel_parameters": params if args.assignments is None else None,
        "temperature": args.temperature if args.assignments is None else None,
        "neighborhood_radius": args.radius,
        "curvature_edges": edges.tolist(),
        "c_min": args.c_min,
        "curvature_consistency": "mutual curvature-class membership, initialized by an ungated support pass",
    }
    ui = SupportUI(original, probabilities, angles, support, classes, output, metadata)
    if args.save_only:
        saved = ui.save()
        plt.close(ui.fig)
        if not saved:
            parser.exit(1, ui.status.get_text() + "\n")
    else:
        plt.show()


if __name__ == "__main__":
    main()
