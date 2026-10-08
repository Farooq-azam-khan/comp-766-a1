import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, CheckButtons, Slider
from PIL import Image

from convolution import (
    ORIENTATION_COUNT,
    X_RANGE,
    Y_RANGE,
    KernelParameters,
    OrientationResult,
    convolve_image_with_all_orientations,
    save_greyscale_img,
    softmax_results,
    tangent_angles,
    tangent_segments,
)

ButtonVariant = Literal["primary", "secondary", "accent", "outline"]


@dataclass(frozen=True)
class ButtonStyle:
    background: str
    foreground: str
    hover: str
    pressed: str
    border: str


BUTTON_STYLES: dict[ButtonVariant, ButtonStyle] = {
    "primary": ButtonStyle("#18181b", "#ffffff", "#303036", "#09090b", "#18181b"),
    "secondary": ButtonStyle("#f4f4f5", "#27272a", "#e4e4e7", "#d4d4d8", "#e4e4e7"),
    "accent": ButtonStyle("#2563eb", "#ffffff", "#1d4ed8", "#1e40af", "#2563eb"),
    "outline": ButtonStyle("#ffffff", "#27272a", "#f4f4f5", "#e4e4e7", "#d4d4d8"),
}


class VariantButton(Button):
    """A Matplotlib button with shared colors and rounded interaction states."""

    def __init__(self, ax: Axes, label: str, variant: ButtonVariant) -> None:
        # Keep the native button's hit testing and callbacks. Its background is
        # transparent so the rounded patch supplies all visible button states.
        super().__init__(ax, label, color="none", hovercolor="none", useblit=False)
        self.style = BUTTON_STYLES[variant]
        self._pressed = False
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.background = FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=self.style.background,
            edgecolor=self.style.border,
            linewidth=0.8,
            clip_on=False,
            zorder=1,
        )
        ax.add_patch(self.background)
        self.label.set(color=self.style.foreground, fontsize=10, fontweight="bold")
        self._resize(None)
        self.connect_event("resize_event", self._resize)
        for name in ("motion_notify_event", "button_press_event", "button_release_event"):
            self.connect_event(name, self._update_state)

    def _resize(self, _event: object) -> None:
        # Match the corner radius in physical units on wide and narrow buttons.
        aspect = self.ax.bbox.width / self.ax.bbox.height
        self.background.set_mutation_aspect(aspect)
        self.background.set_boxstyle("round", pad=0, rounding_size=0.16 / aspect)

    def _update_state(self, event: MouseEvent) -> None:
        if self.ignore(event):
            return
        inside = self.ax.contains(event)[0]
        if event.name == "button_press_event":
            self._pressed = inside and self.canvas.mouse_grabber is self.ax
        elif event.name == "button_release_event":
            self._pressed = False
        color = self.style.background
        if inside:
            color = self.style.pressed if self._pressed else self.style.hover
        # Skip redraws for mouse motion that leaves the state unchanged.
        if self.background.get_facecolor() == to_rgba(color):
            return
        self.background.set_facecolor(color)
        if self.drawon:
            self.canvas.draw_idle()


def angle_label(index: int) -> str:
    """Label angles relative to the vertical reference kernel."""
    return f"{index * 180 / ORIENTATION_COUNT:g}°"

@final
class ConvolutionUI:
    def __init__(self, path: Path) -> None:
        self.path = path
        with Image.open(path) as source:
            color = source.convert("RGB")
            self.original = np.asarray(color, dtype=np.uint8)
            self.image = np.asarray(color.convert("L"), dtype=np.float64) / 255.0

        self.results: list[OrientationResult] = []
        self.probabilities = np.empty((0, 0, 0), dtype=np.float64)
        self.vector_segments = np.empty((0, 2, 2), dtype=np.float64)
        self.applied_params: KernelParameters | None = None
        self.selected_index: int | None = 0
        self.dirty = True

        self.create_figure()
        self.create_controls()
        self.create_timer()
        self.apply()

    def create_figure(self) -> None:
        self.fig, axes = plt.subplots(1, 4, figsize=(18, 9))
        self.fig.subplots_adjust(
            left=0.03, right=0.98, top=0.90, bottom=0.48, wspace=0.14
        )
        self.fig.suptitle(f"Oriented convolution — {self.path.name}", fontsize=16)
        self.original_ax, gray_ax, self.kernel_ax, self.response_ax = axes
        self.original_ax.imshow(self.original)
        self.vector_artist = LineCollection(
            [], colors="#2563eb", linewidths=0.9, alpha=0.95
        )
        self.original_ax.add_collection(self.vector_artist)
        self.original_ax.set_title("Initial tangents on original")
        gray_ax.imshow(self.image, cmap="gray", vmin=0, vmax=1)
        gray_ax.set_title("Grayscale input")
        self.kernel_artist = self.kernel_ax.imshow(
            np.zeros((3, 3)), cmap="RdBu_r", vmin=-1, vmax=1
        )
        self.response_artist = self.response_ax.imshow(
            self.image, cmap="gray", vmin=0, vmax=1
        )
        for ax in axes:
            ax.axis("off")

        self.kernel_grid_axes = []
        self.response_grid_axes = []
        self.kernel_grid_artists = []
        self.response_grid_artists = []
        for panel, grid_axes, grid_artists, color_map in (
            (self.kernel_ax, self.kernel_grid_axes, self.kernel_grid_artists, "RdBu_r"),
            (
                self.response_ax,
                self.response_grid_axes,
                self.response_grid_artists,
                "gray",
            ),
        ):
            bounds = panel.get_position()
            gap = 0.004
            tile_width = (bounds.width - 3 * gap) / 4
            tile_height = (bounds.height - 3 * gap) / 4
            for index in range(ORIENTATION_COUNT):
                row, col = divmod(index, 4)
                tile_ax = self.fig.add_axes(
                    (
                        bounds.x0 + col * (tile_width + gap),
                        bounds.y0 + (3 - row) * (tile_height + gap),
                        tile_width,
                        tile_height,
                    )
                )
                tile_ax.axis("off")
                tile_ax.text(
                    0.03,
                    0.97,
                    angle_label(index),
                    transform=tile_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
                )
                artist = tile_ax.imshow(np.zeros((3, 3)), cmap=color_map)
                tile_ax.set_visible(False)
                grid_axes.append(tile_ax)
                grid_artists.append(artist)

        self.kernel_header = self.fig.text(
            self.kernel_ax.get_position().x0, 0.92, "Kernels", fontsize=11
        )
        self.response_header = self.fig.text(
            self.response_ax.get_position().x0, 0.92, "Convolved images", fontsize=11
        )
        self.kernel_header.set_visible(False)
        self.response_header.set_visible(False)
        self.status = self.fig.text(0.03, 0.445, "", fontsize=10)

    def create_controls(self) -> None:
        # Slider values use the same names as KernelParameters.
        specifications = [
            ("sigma_y", 0.05, 2.5, 1.0, 0.01),
            ("sigma_1", 0.05, 2.0, 0.8, 0.01),
            ("sigma_2", 0.05, 2.0, 0.3, 0.01),
            ("sigma_3", 0.05, 2.0, 0.8, 0.01),
            ("A", 0.0, 3.0, 0.5, 0.01),
            ("B", 0.0, 3.0, 1.5, 0.01),
            ("C", 0.0, 3.0, 0.5, 0.01),
            ("samples", 5, 201, 101, 2),
        ]
        self.sliders: dict[str, Slider] = {}
        for index, (name, low, high, initial, step) in enumerate(specifications):
            col, row = divmod(index, 4)
            slider_ax = self.fig.add_axes(
                (0.12 + col * 0.47, 0.365 - row * 0.064, 0.31, 0.026)
            )
            slider = Slider(
                slider_ax,
                name,
                low,
                high,
                valinit=initial,
                valstep=step,
                valfmt="%d" if name == "samples" else "%.2f",
            )
            slider.on_changed(self.schedule_update)
            self.sliders[name] = slider

        self.vector_threshold = Slider(
            self.fig.add_axes((0.12, 0.405, 0.18, 0.022)),
            "Vector threshold", 0.0, 1.0, valinit=0.2, valstep=0.01, valfmt="%.2f",
        )
        self.vector_threshold.on_changed(self.refresh_vectors)
        self.softmax_temperature = Slider(
            self.fig.add_axes((0.49, 0.405, 0.12, 0.022)),
            "Softmax temp", 0.001, 0.1, valinit=0.02, valstep=0.001, valfmt="%.3f",
        )
        self.softmax_temperature.on_changed(self.refresh_assignments)
        self.vector_spacing = Slider(
            self.fig.add_axes((0.12, 0.132, 0.31, 0.018)),
            "Vector spacing", 1, 16, valinit=4, valstep=1, valfmt="%d px",
        )
        self.vector_spacing.on_changed(self.refresh_vectors)

        self.checks = CheckButtons(
            self.fig.add_axes((0.05, 0.025, 0.23, 0.105)),
            ["Auto update", "Auto contrast"],
            [True, True],
        )
        self.checks.on_clicked(self.toggle_option)

        self.orientation_labels = ["All"] + [
            angle_label(i) for i in range(ORIENTATION_COUNT)
        ]
        self.orientation_button = VariantButton(
            self.fig.add_axes((0.72, 0.398, 0.25, 0.047)),
            f"Orientation: {self.orientation_labels[1]} ▾",
            "outline",
        )
        self.orientation_button.on_clicked(self.toggle_orientation_menu)
        self.orientation_menu_ax = self.fig.add_axes((0.73, 0.45, 0.22, 0.45))
        self.orientation_menu_ax.set_zorder(20)
        self.orientation_menu_ax.set(
            xlim=(0, 1), ylim=(0, len(self.orientation_labels))
        )
        self.orientation_menu_ax.set_xticks([])
        self.orientation_menu_ax.set_yticks([])
        self.orientation_menu_ax.set_facecolor("white")
        for spine in self.orientation_menu_ax.spines.values():
            spine.set_color(BUTTON_STYLES["outline"].border)
            spine.set_linewidth(0.8)
        self.orientation_menu_selection = Rectangle(
            (0, len(self.orientation_labels) - 2),
            1,
            1,
            facecolor="#dbeafe",
            edgecolor="none",
        )
        self.orientation_menu_ax.add_patch(self.orientation_menu_selection)
        for row, label in enumerate(self.orientation_labels):
            self.orientation_menu_ax.text(
                0.08,
                len(self.orientation_labels) - row - 0.5,
                label,
                va="center",
                fontsize=10,
                color=BUTTON_STYLES["outline"].foreground,
            )
        self.fig.canvas.mpl_connect(
            "button_press_event", self.on_orientation_menu_click
        )
        self.orientation_menu_ax.set_visible(False)

        self.buttons: list[VariantButton] = []
        for label, x, callback, variant in (
            ("Apply", 0.34, self.apply, "primary"),
            ("Reset", 0.51, self.reset, "secondary"),
            ("Save view", 0.68, self.save, "accent"),
        ):
            button = VariantButton(
                self.fig.add_axes((x, 0.065, 0.14, 0.047)), label, variant
            )
            button.on_clicked(callback)
            self.buttons.append(button)

    def create_timer(self) -> None:
        # Wait until slider motion pauses before computing the 16 responses.
        self.timer = self.fig.canvas.new_timer(interval=350)
        self.timer.single_shot = True
        self.timer.add_callback(self.apply)
        self.fig.canvas.mpl_connect("close_event", lambda event: self.timer.stop())

    def parameters(self) -> KernelParameters:
        return {
            "samples": int(self.sliders["samples"].val),
            "sigma_y": float(self.sliders["sigma_y"].val),
            "sigma_1": float(self.sliders["sigma_1"].val),
            "sigma_2": float(self.sliders["sigma_2"].val),
            "sigma_3": float(self.sliders["sigma_3"].val),
            "A": float(self.sliders["A"].val),
            "B": float(self.sliders["B"].val),
            "C": float(self.sliders["C"].val),
        }

    def message(self, text: str, error: bool = False) -> None:
        self.status.set_text(text)
        self.status.set_color("firebrick" if error else "black")
        self.fig.canvas.draw_idle()

    def schedule_update(self, _value: float) -> None:
        self.dirty = True
        self.timer.stop()
        self.message("Values changed. Click Apply or pause dragging to update.")
        if self.checks.get_status()[0]:
            self.timer.start()

    def toggle_option(self, label: str | None) -> None:
        if label == "Auto update":
            self.timer.stop()
            if self.checks.get_status()[0] and self.dirty:
                self.timer.start()
        elif label == "Auto contrast":
            self.refresh_display()

    def toggle_orientation_menu(self, _event: object = None) -> None:
        self.orientation_menu_ax.set_visible(not self.orientation_menu_ax.get_visible())
        self.fig.canvas.draw_idle()

    def on_orientation_menu_click(self, event: MouseEvent) -> None:
        if event.inaxes is not self.orientation_menu_ax or event.ydata is None:
            return
        row = len(self.orientation_labels) - 1 - int(event.ydata)
        if 0 <= row < len(self.orientation_labels):
            self.select_orientation(self.orientation_labels[row])

    def select_orientation(self, label: str | None) -> None:
        if label is None:
            return
        row = self.orientation_labels.index(label)
        self.selected_index = None if row == 0 else row - 1
        self.orientation_menu_selection.set_y(len(self.orientation_labels) - row - 1)
        self.orientation_button.label.set_text(f"Orientation: {label} ▾")
        self.orientation_menu_ax.set_visible(False)
        self.refresh_display()

    def apply(self, _event: object = None) -> None:
        self.timer.stop()
        params = self.parameters()
        try:
            results = convolve_image_with_all_orientations(self.image, params)
        except ValueError as exc:
            self.message(f"{exc} Previous previews kept; saving is disabled.", True)
            return
        self.results = results
        self.refresh_assignments()
        self.applied_params = params
        self.dirty = False
        self.refresh_display()
        self.message(
            f"Applied {ORIENTATION_COUNT} orientations with a {params['samples']} × {params['samples']} kernel."
        )

    def refresh_assignments(self, _value: float = 0.0) -> None:
        if not self.results:
            return
        self.probabilities = softmax_results(
            self.results, temperature=float(self.softmax_temperature.val)
        )
        self.refresh_vectors()

    def refresh_vectors(self, _value: float = 0.0) -> None:
        if not self.results:
            return
        threshold = float(self.vector_threshold.val)
        spacing = int(self.vector_spacing.val)
        self.vector_segments = tangent_segments(
            tangent_angles(np.array([item.angle_radians for item in self.results])),
            self.probabilities, threshold, spacing,
        )
        self.vector_artist.set_segments(self.vector_segments)
        self.original_ax.set_title(
            f"Initial tangents · p > {threshold:.2f}\n"
            f"{len(self.vector_segments):,} vectors · every {spacing} px",
            fontsize=10,
        )
        if not len(self.vector_segments):
            maximum = float(self.probabilities[:, ::spacing, ::spacing].max())
            self.original_ax.set_title(
                f"Initial tangents · p > {threshold:.2f}\n"
                f"Max p = {maximum:.3f}; lower threshold or temp", fontsize=10,
            )
        self.fig.canvas.draw_idle()

    def display_limits(self) -> tuple[float, float]:
        if not self.results or not self.checks.get_status()[1]:
            return 0.0, 1.0
        samples = []
        for item in self.results:
            height, width = item.response.shape
            samples.append(
                item.response[:: max(1, height // 128), :: max(1, width // 128)].ravel()
            )
        low, high = np.percentile(np.concatenate(samples), [1, 99])
        if high - low < 1e-12:
            return 0.0, 1.0
        return float(low), float(high)

    def refresh_display(self) -> None:
        if not self.results:
            return
        low, high = self.display_limits()
        kernel_limit = max(float(np.abs(item.kernel).max()) for item in self.results)
        kernel_limit = max(kernel_limit, 1e-12)
        for item, kernel_artist, response_artist in zip(
            self.results, self.kernel_grid_artists, self.response_grid_artists
        ):
            kernel_artist.set_data(item.kernel)
            kernel_artist.set_clim(-kernel_limit, kernel_limit)
            height, width = item.response.shape
            response_artist.set_data(
                item.response[
                    :: max(1, (height + 255) // 256),
                    :: max(1, (width + 255) // 256),
                ]
            )
            response_artist.set_clim(low, high)

        show_all = self.selected_index is None
        self.kernel_ax.set_visible(not show_all)
        self.response_ax.set_visible(not show_all)
        self.kernel_header.set_visible(show_all)
        self.response_header.set_visible(show_all)
        for ax in self.kernel_grid_axes + self.response_grid_axes:
            ax.set_visible(show_all)

        if not show_all and self.selected_index is not None:
            item = self.results[self.selected_index]
            self.kernel_artist.set_data(item.kernel)
            self.kernel_artist.set_clim(-kernel_limit, kernel_limit)
            self.kernel_artist.set_extent((*X_RANGE, Y_RANGE[1], Y_RANGE[0]))
            self.response_artist.set_data(item.response)
            self.response_artist.set_clim(low, high)
            self.kernel_ax.set_title(f"Kernel · {angle_label(item.index)}")
            self.response_ax.set_title(f"Convolved · {angle_label(item.index)}")
        self.fig.canvas.draw_idle()

    def reset(self, _event: object = None) -> None:
        for slider in self.sliders.values():
            slider.reset()
        self.vector_threshold.reset()
        self.softmax_temperature.reset()
        self.vector_spacing.reset()
        self.apply()

    def save_vector_overlay(self, path: Path) -> None:
        height, width = self.image.shape
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(self.original)
        ax.add_collection(LineCollection(
            self.vector_segments, colors="#2563eb", linewidths=0.9, alpha=0.95
        ))
        ax.axis("off")
        try:
            fig.savefig(path, dpi=100)
        finally:
            plt.close(fig)

    def save_contact_sheet(
        self, kind: str, path: Path, low: float, high: float, kernel_limit: float
    ) -> None:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for ax, item in zip(axes.flat, self.results):
            if kind == "kernels":
                ax.imshow(
                    item.kernel, cmap="RdBu_r", vmin=-kernel_limit, vmax=kernel_limit
                )
            else:
                ax.imshow(item.response, cmap="gray", vmin=low, vmax=high)
            ax.set_title(angle_label(item.index), fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def save(self, _event: object = None) -> None:
        if self.dirty:
            self.apply()
        params = self.applied_params
        if self.dirty or not self.results or params is None:
            return

        indices = (
            range(ORIENTATION_COUNT)
            if self.selected_index is None
            else (self.selected_index,)
        )
        suffix = "all" if self.selected_index is None else f"{self.selected_index:02d}"
        output_root = self.path.with_name(f"{self.path.stem}_orientations")
        run_name = f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%fZ')}_{suffix}"
        output_dir = output_root / run_name
        low, high = self.display_limits()
        kernel_limit = max(
            1e-12, *(float(np.abs(item.kernel).max()) for item in self.results)
        )
        orientations = []

        try:
            output_dir.mkdir(parents=True)
            Image.fromarray(self.original).save(output_dir / "original.png")
            save_greyscale_img(self.image, output_dir / "grayscale.png")
            self.save_vector_overlay(output_dir / "initial_tangents.png")
            np.save(output_dir / "assignments.npy", self.probabilities)
            for index in indices:
                item = self.results[index]
                name = (
                    f"orientation_{index:02d}_{np.degrees(item.angle_radians):06.2f}deg"
                )
                kernel_colors = np.clip(
                    0.5 + item.kernel / (2 * kernel_limit), 0.0, 1.0
                )
                rgba = np.asarray(colormaps["RdBu_r"](kernel_colors, bytes=True))
                Image.fromarray(rgba[..., :3]).save(output_dir / f"{name}_kernel.png")
                display_response = (item.response - low) / (high - low)
                save_greyscale_img(
                    display_response, output_dir / f"{name}_convolved.png"
                )
                np.save(output_dir / f"{name}_kernel.npy", item.kernel)
                np.save(output_dir / f"{name}_response.npy", item.response)
                orientations.append(
                    {
                        "index": index,
                        "angle_degrees": float(np.degrees(item.angle_radians)),
                    }
                )

            if self.selected_index is None:
                self.save_contact_sheet(
                    "kernels", output_dir / "kernels_all.png", low, high, kernel_limit
                )
                self.save_contact_sheet(
                    "responses",
                    output_dir / "convolved_all.png",
                    low,
                    high,
                    kernel_limit,
                )
            manifest = {
                "source": str(self.path),
                "parameters": params,
                "orientations": orientations,
                "angle_reference": "0 degrees is the vertical kernel",
                "display_range": [low, high],
                "auto_contrast": bool(self.checks.get_status()[1]),
                "kernel_normalization": "sum to one",
                "image_padding": "symmetric",
                "initial_assignments": {
                    "normalization": "softmax across all 16 orientations at each pixel",
                    "temperature": float(self.softmax_temperature.val),
                    "vector_threshold": float(self.vector_threshold.val),
                    "threshold_comparison": "strictly greater than",
                    "vector_spacing_pixels": int(self.vector_spacing.val),
                    "vectors_drawn": len(self.vector_segments),
                    "overlay": "initial_tangents.png",
                    "probabilities": "assignments.npy",
                    "probability_axes": ["orientation", "row", "column"],
                },
                "raw_data": "Matching *_kernel.npy and *_response.npy files",
            }
            (output_dir / f"manifest_{suffix}.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        except (OSError, ValueError) as exc:
            self.message(f"Save failed: {exc}", True)
            return
        self.message(
            f"Saved {len(orientations)} orientation(s) to {output_root.name}/{run_name}."
        )
