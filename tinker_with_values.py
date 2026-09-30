"""Run: python convolution_ui.py [path/to/image.png]"""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import fftconvolve


def G_np(
    xs: NDArray[np.float64], ys: NDArray[np.float64],
    sigma_y: float = 1.0, sigma_1: float = 0.8,
    sigma_2: float = 0.3, sigma_3: float = 0.8,
    A: float = 0.5, B: float = 1.5, C: float = 0.5,
) -> NDArray[np.float64]:
    lsf = (A * np.exp(-(xs / sigma_1)**2)
           - B * np.exp(-(xs / sigma_2)**2)
           + C * np.exp(-(xs / sigma_3)**2))
    return lsf * np.exp(-(ys / sigma_y)**2)


def generate_kernel(params):
    samples = int(params['samples'])
    X, Y = np.meshgrid(np.linspace(-2, 2, samples),
                       np.linspace(-2.5, 2.5, samples))
    kernel = G_np(X, Y, **{k: v for k, v in params.items() if k != 'samples'})
    total = kernel.sum()
    if abs(total) <= 1e-6 * max(np.abs(kernel).sum(), 1e-12):
        raise ValueError('Kernel sum is near zero. Adjust amplitudes or widths.')
    return kernel / total


def convolve_image_with_kernel(kernel, image):
    py, px = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((py, py), (px, px)), mode='symmetric')
    return fftconvolve(padded, kernel, mode='valid')


class ConvolutionUI:
    def __init__(self, path):
        self.path = Path(path)
        with Image.open(self.path) as source:
            self.image = np.asarray(source.convert('L'), dtype=np.float64) / 255
        self.result = None
        self.applied_params = None
        self.dirty = True
        self.fig, axes = plt.subplots(1, 3, figsize=(14, 8))
        self.fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.48,
                                 wspace=0.15)
        self.fig.suptitle(f'Gaussian convolution — {self.path.name}')
        axes[0].imshow(self.image, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original')
        self.kernel_artist = axes[1].imshow(np.zeros((101, 101)),
                                             cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title('Normalized kernel')
        self.result_artist = axes[2].imshow(self.image, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Convolved')
        for ax in axes:
            ax.axis('off')
        self.status = self.fig.text(0.05, 0.435, '', fontsize=10)

        # Each slider's bounds can be changed here.
        specifications = [
            ('sigma_y', 0.05, 2.5, 1.0, 0.01),
            ('sigma_1', 0.05, 2.0, 0.8, 0.01),
            ('sigma_2', 0.05, 2.0, 0.3, 0.01),
            ('sigma_3', 0.05, 2.0, 0.8, 0.01),
            ('A', 0.0, 3.0, 0.5, 0.01),
            ('B', 0.0, 3.0, 1.5, 0.01),
            ('C', 0.0, 3.0, 0.5, 0.01),
            ('samples', 5, 201, 101, 2),
        ]
        self.sliders = {}
        for i, (name, low, high, initial, step) in enumerate(specifications):
            col, row = i // 4, i % 4
            ax = self.fig.add_axes([0.12 + col * 0.47, 0.365 - row * 0.064,
                                    0.31, 0.026])
            slider = Slider(ax, name, low, high, valinit=initial, valstep=step,
                            valfmt='%d' if name == 'samples' else '%.2f')
            slider.on_changed(self.schedule_update)
            self.sliders[name] = slider

        self.checks = CheckButtons(self.fig.add_axes([0.05, 0.025, 0.23, 0.105]),
                                  ['Auto update', 'Auto contrast'], [True, False])
        self.checks.on_clicked(self.toggle_option)
        self.buttons = []
        for label, x, callback in [('Apply', 0.34, self.apply),
                                    ('Reset', 0.51, self.reset),
                                    ('Save PNG', 0.68, self.save)]:
            button = Button(self.fig.add_axes([x, 0.065, 0.14, 0.045]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        # Wait until slider motion pauses before computing another FFT.
        self.timer = self.fig.canvas.new_timer(interval=350)
        self.timer.single_shot = True
        self.timer.add_callback(self.apply)
        self.fig.canvas.mpl_connect('close_event', lambda event: self.timer.stop())
        self.apply()

    def parameters(self):
        return {k: int(s.val) if k == 'samples' else float(s.val)
                for k, s in self.sliders.items()}

    def message(self, text, error=False):
        self.status.set_text(text)
        self.status.set_color('firebrick' if error else 'black')
        self.fig.canvas.draw_idle()

    def schedule_update(self, value=None):
        self.dirty = True
        self.timer.stop()
        self.message('Values changed — preview pending. Click Apply or pause dragging.')
        if self.checks.get_status()[0]:
            self.timer.start()

    def toggle_option(self, label):
        if label == 'Auto update':
            self.timer.stop()
            if self.checks.get_status()[0] and self.dirty:
                self.timer.start()
        else:
            self.refresh_display()

    def apply(self, event=None):
        self.timer.stop()
        params = self.parameters()
        try:
            kernel = generate_kernel(params)
            result = convolve_image_with_kernel(kernel, self.image)
        except ValueError as exc:
            self.message(f'{exc} Previous preview kept; saving is disabled.', True)
            return
        self.result = result
        self.applied_params = params
        self.dirty = False
        self.kernel_artist.set_data(kernel)
        self.kernel_artist.set_extent((-2, 2, 2.5, -2.5))
        limit = max(float(np.abs(kernel).max()), 1e-12)
        self.kernel_artist.set_clim(-limit, limit)
        self.refresh_display()
        self.message(f"Applied {params['samples']} × {params['samples']} kernel. "
                     f'Raw output range: {result.min():.3f} to {result.max():.3f}')

    def displayed_image(self):
        if self.checks.get_status()[1]:
            low, high = np.percentile(self.result, [1, 99])
            if high > low:
                return np.clip((self.result - low) / (high - low), 0, 1)
        return np.clip(self.result, 0, 1)

    def refresh_display(self):
        if self.result is not None:
            self.result_artist.set_data(self.displayed_image())
            self.fig.canvas.draw_idle()

    def reset(self, event=None):
        for slider in self.sliders.values():
            slider.reset()
        self.apply()

    def save(self, event=None):
        # Always apply current sliders before exporting, including pending edits.
        if self.dirty:
            self.apply()
        if self.dirty or self.result is None:
            return
        output = self.path.with_name(f'{self.path.stem}_convolved.png')
        settings = output.with_suffix('.json')
        try:
            Image.fromarray(np.round(self.displayed_image() * 255).astype(np.uint8)).save(output)
            settings.write_text(json.dumps({
                'source': str(self.path), **self.applied_params,
                'x_range': [-2, 2], 'y_range': [-2.5, 2.5],
                'normalization': 'sum', 'boundary': 'symmetric',
                'auto_contrast': bool(self.checks.get_status()[1]),
            }, indent=2) + '\n', encoding='utf-8')
        except OSError as exc:
            self.message(f'Save failed: {exc}', True)
            return
        self.message(f'Saved {output.name} and {settings.name} beside the input image.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', type=Path,
                        default=Path('curves/spaghetti.png'))
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f'Image not found: {args.image}')
    ui = ConvolutionUI(args.image)  # Keep widgets alive while the window is open.
    plt.show()


if __name__ == '__main__':
    main()
