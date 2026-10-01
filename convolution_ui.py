"""Matplotlib controls and previews for the Gaussian convolution."""

import json
from pathlib import Path
from typing import final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider
from numpy.typing import NDArray

from convolution import (
    X_RANGE,
    Y_RANGE,
    KernelParameters,
    convolve_image_with_kernel,
    generate_kernel,
    load_greyscale_img,
    save_greyscale_img,
)


@final
class ConvolutionUI:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.image = load_greyscale_img(self.path)
        self.result: NDArray[np.float64] | None = None
        self.applied_params: KernelParameters | None = None
        self.dirty = True

        self.create_figure()
        self.create_controls()
        self.create_timer()
        self.apply()

    def create_figure(self) -> None:
        self.fig, axes = plt.subplots(1, 3, figsize=(14, 8))
        self.fig.subplots_adjust(
            left=0.04, right=0.96, top=0.92, bottom=0.48, wspace=0.15
        )
        self.fig.suptitle(f"Gaussian convolution — {self.path.name}")
        axes[0].imshow(self.image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original")
        self.kernel_artist = axes[1].imshow(
            np.zeros((101, 101)), cmap="RdBu_r", vmin=-1, vmax=1
        )
        axes[1].set_title("Normalized kernel")
        self.result_artist = axes[2].imshow(self.image, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Convolved")
        for ax in axes:
            ax.axis("off")
        self.status = self.fig.text(0.05, 0.435, "", fontsize=10)

    def create_controls(self) -> None:
        # Each slider's bounds can be changed here.
        specifications = [
            ("sigma_y", 0.05, 2.5, 1.0, 0.01),
            ("sigma_1", 0.05, 2.0, 0.8, 0.01),
            ("sigma_2", 0.05, 2.0, 0.3, 0.01),
            ("sigma_3", 0.05, 2.0, 0.8, 0.01),
            ("A", 0.0, 3.0, 0.5, 0.01),
            ("B", 0.0, 3.0, 1.5, 0.01),
            ("C", 0.0, 3.0, 0.5, 0.01),
            ("samples", 5, 201, 100, 2),
        ]
        self.sliders: dict[str, Slider] = {}
        for i, (name, low, high, initial, step) in enumerate(specifications):
            col, row = i // 4, i % 4
            ax = self.fig.add_axes(
                (0.12 + col * 0.47, 0.365 - row * 0.064, 0.31, 0.026)
            )
            slider = Slider(
                ax,
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
        self.buttons: list[Button] = []
        for label, x, callback in [
            ("Apply", 0.34, self.apply),
            ("Reset", 0.51, self.reset),
            ("Save PNG", 0.68, self.save),
        ]:
            button = Button(self.fig.add_axes((x, 0.065, 0.14, 0.045)), label)
            button.on_clicked(callback)
            self.buttons.append(button)

    def create_timer(self) -> None:
        # Wait until slider motion pauses before computing another FFT.
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
        self.message("Values changed — preview pending. Click Apply or pause dragging.")
        if self.checks.get_status()[0]:
            self.timer.start()

    def toggle_option(self, label: str | None) -> None:
        if label == "Auto update":
            self.timer.stop()
            if self.checks.get_status()[0] and self.dirty:
                self.timer.start()
        elif label == "Auto contrast":
            self.refresh_display()

    def apply(self, _event: object = None) -> None:
        self.timer.stop()
        params = self.parameters()
        try:
            kernel = generate_kernel(**params)
            result = convolve_image_with_kernel(kernel, self.image)
        except ValueError as exc:
            self.message(f"{exc} Previous preview kept; saving is disabled.", True)
            return
        self.result = result
        self.applied_params = params
        self.dirty = False
        self.kernel_artist.set_data(kernel)
        self.kernel_artist.set_extent((*X_RANGE, Y_RANGE[1], Y_RANGE[0]))
        limit = max(float(np.abs(kernel).max()), 1e-12)
        self.kernel_artist.set_clim(-limit, limit)
        self.refresh_display()
        size = params["samples"]
        summary = (
            f"Applied {size} × {size} kernel. Raw output range: "
            + f"{result.min():.3f} to {result.max():.3f}"
        )
        self.message(summary)

    def displayed_image(self) -> NDArray[np.float64]:
        result = self.result
        if result is None:
            raise RuntimeError("No convolution result is available.")
        if self.checks.get_status()[1]:
            low, high = np.percentile(result, [1, 99])
            if high > low:
                return np.clip((result - low) / (high - low), 0, 1)
        return np.clip(result, 0, 1)

    def refresh_display(self) -> None:
        if self.result is not None:
            self.result_artist.set_data(self.displayed_image())
            self.fig.canvas.draw_idle()

    def reset(self, _event: object = None) -> None:
        for slider in self.sliders.values():
            slider.reset()
        self.apply()

    def save(self, _event: object = None) -> None:
        # Always apply current sliders before exporting, including pending edits.
        if self.dirty:
            self.apply()
        params = self.applied_params
        if self.dirty or self.result is None or params is None:
            return
        output = self.path.with_name(f"{self.path.stem}_convolved.png")
        settings = output.with_suffix(".json")
        try:
            save_greyscale_img(self.displayed_image(), output)
            settings.write_text(
                json.dumps(
                    {
                        "source": str(self.path),
                        **params,
                        "x_range": list(X_RANGE),
                        "y_range": list(Y_RANGE),
                        "normalization": "sum",
                        "boundary": "symmetric",
                        "auto_contrast": bool(self.checks.get_status()[1]),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            self.message(f"Save failed: {exc}", True)
            return
        self.message(f"Saved {output.name} and {settings.name} beside the input image.")
