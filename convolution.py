"""Generate a Gaussian difference kernel and apply it to a grayscale image."""

from os import PathLike
from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import fftconvolve

X_RANGE = (-2.0, 2.0)
Y_RANGE = (-2.5, 2.5)


class KernelParameters(TypedDict):
    samples: int
    sigma_y: float
    sigma_1: float
    sigma_2: float
    sigma_3: float
    A: float
    B: float
    C: float


def G_np(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    sigma_y: float = 1.0,
    sigma_1: float = 0.8,
    sigma_2: float = 0.3,
    sigma_3: float = 0.8,
    A: float = 0.5,
    B: float = 1.5,
    C: float = 0.5,
) -> NDArray[np.float64]:
    """Evaluate the assignment's Gaussian difference function on a grid."""
    line_spread = (
        A * np.exp(-((xs / sigma_1) ** 2))
        - B * np.exp(-((xs / sigma_2) ** 2))
        + C * np.exp(-((xs / sigma_3) ** 2))
    )
    return line_spread * np.exp(-((ys / sigma_y) ** 2))

def generate_grid_from_angle(theta, samples: int = 15):
    xs, ys = np.meshgrid(
            np.linspace(*X_RANGE, samples),
            np.linspace(*Y_RANGE, samples),
        )
    return xs*np.cos(theta)+ys*np.sin(theta), -xs*np.sin(theta)+ys*np.cos(theta)

def generate_kernel(
    theta=0*np.pi/16,
    samples: int = 100,
    *,
    sigma_y: float = 1.0,
    sigma_1: float = 0.8,
    sigma_2: float = 0.3,
    sigma_3: float = 0.8,
    A: float = 0.5,
    B: float = 1.5,
    C: float = 0.5,
) -> NDArray[np.float64]:
    """Sample G and normalize its sum to one for image convolution."""
    if samples < 2:
        raise ValueError("The kernel needs at least two samples per axis.")
    xs, ys = generate_grid_from_angle(theta,samples)
    kernel = G_np(xs, ys, sigma_y, sigma_1, sigma_2, sigma_3, A, B, C)
    total = kernel.sum()
    if abs(total) <= 1e-6 * max(np.abs(kernel).sum(), 1e-12):
        raise ValueError("Kernel sum is near zero. Adjust amplitudes or widths.")
    return kernel / total


def convolve_image_with_kernel(
    kernel: NDArray[np.float64], image: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convolve an image with symmetric padding and retain its original size."""
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="symmetric")
    return cast(NDArray[np.float64], fftconvolve(padded, kernel, mode="valid"))

def convolve_image_with_all_orientations(image):
    kernels = [generate_kernel(i*np.pi/16) for i in range(0,15)]
    images = [convolve_image_with_kernel(kernel, image) for kernel in kernels]

def load_greyscale_img(path: str | PathLike[str]) -> NDArray[np.float64]:
    """Read an image as grayscale values between zero and one."""
    with Image.open(path) as source:
        return np.asarray(source.convert("L"), dtype=np.float64) / 255.0


def save_greyscale_img(image: NDArray[np.float64], path: str | PathLike[str]) -> None:
    """Save grayscale values as an 8-bit PNG."""
    pixels = np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def visualize_convolution_surface(kernel: NDArray[np.float64]) -> None:
    """Show the sampled kernel as a 3D surface."""
    xs, ys = np.meshgrid(
        np.linspace(*X_RANGE, kernel.shape[1]),
        np.linspace(*Y_RANGE, kernel.shape[0]),
    )
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xs, ys, kernel, cmap="viridis", edgecolor="none")
    ax.set(xlim=X_RANGE, ylim=Y_RANGE, xlabel="X", ylabel="Y", zlabel="G(x, y)")
    ax.view_init(elev=20, azim=-35)
    plt.tight_layout()
    plt.show()


def main() -> None:
    curves_dir = Path("curves")
    image_path = curves_dir / "fingerprint.png"
    image = load_greyscale_img(image_path)
    kernel = generate_kernel()
    result = convolve_image_with_kernel(kernel, image)
    save_greyscale_img(result, curves_dir / f"{image_path.stem}_convolved.png")

    # visualize_convolution_surface(kernel)
    _, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[1].imshow(kernel, cmap="RdBu_r")
    axes[1].set_title("Kernel")
    axes[2].imshow(result, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Convolved")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
