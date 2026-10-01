"""Matplotlib controls and previews for the oriented convolution filters."""

import json
from pathlib import Path
from typing import final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from PIL import Image

from convolution import (
    ORIENTATION_COUNT,
    X_RANGE,
    Y_RANGE,
    KernelParameters,
    OrientationResult,
    convolve_image_with_all_orientations,
    save_greyscale_img,
)


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
        original_ax, gray_ax, self.kernel_ax, self.response_ax = axes
        original_ax.imshow(self.original)
        original_ax.set_title("Original color")
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

        self.checks = CheckButtons(
            self.fig.add_axes((0.05, 0.025, 0.23, 0.105)),
            ["Auto update", "Auto contrast"],
            [True, False],
        )
        self.checks.on_clicked(self.toggle_option)

        labels = ["All"] + [angle_label(i) for i in range(ORIENTATION_COUNT)]
        self.orientation_button = Button(
            self.fig.add_axes((0.72, 0.405, 0.25, 0.04)),
            f"Orientation: {labels[1]} ▾",
        )
        self.orientation_button.on_clicked(self.toggle_orientation_menu)
        self.orientation_menu_ax = self.fig.add_axes((0.73, 0.12, 0.22, 0.68))
        self.orientation_menu_ax.set_zorder(20)
        self.orientation_menu = RadioButtons(self.orientation_menu_ax, labels, active=1)
        self.orientation_menu.on_clicked(self.select_orientation)
        self.orientation_menu_ax.set_visible(False)

        self.buttons: list[Button] = []
        for label, x, callback in (
            ("Apply", 0.34, self.apply),
            ("Reset", 0.51, self.reset),
            ("Save view", 0.68, self.save),
        ):
            button = Button(self.fig.add_axes((x, 0.065, 0.14, 0.045)), label)
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

    def select_orientation(self, label: str | None) -> None:
        if label is None:
            return
        self.selected_index = (
            None
            if label == "All"
            else int(round(float(label.removesuffix("°")) * ORIENTATION_COUNT / 180))
        )
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
        self.applied_params = params
        self.dirty = False
        self.refresh_display()
        self.message(
            f"Applied {ORIENTATION_COUNT} orientations with a {params['samples']} × {params['samples']} kernel."
        )

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
        self.apply()

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

        output_dir = self.path.with_name(f"{self.path.stem}_orientations")
        indices = (
            range(ORIENTATION_COUNT)
            if self.selected_index is None
            else (self.selected_index,)
        )
        suffix = "all" if self.selected_index is None else f"{self.selected_index:02d}"
        low, high = self.display_limits()
        kernel_limit = max(
            1e-12, *(float(np.abs(item.kernel).max()) for item in self.results)
        )
        orientations = []

        try:
            output_dir.mkdir(exist_ok=True)
            Image.fromarray(self.original).save(output_dir / "original.png")
            save_greyscale_img(self.image, output_dir / "grayscale.png")
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
                "raw_data": "Matching *_kernel.npy and *_response.npy files",
            }
            (output_dir / f"manifest_{suffix}.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        except (OSError, ValueError) as exc:
            self.message(f"Save failed: {exc}", True)
            return
        self.message(f"Saved {len(orientations)} orientation(s) to {output_dir}.")
