"""Generate a Gaussian difference kernel and apply it to a grayscale image."""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import fftconvolve
from scipy.special import softmax

X_RANGE = (-2.0, 2.0)
Y_RANGE = (-2.5, 2.5)
ORIENTATION_COUNT = 16


class KernelParameters(TypedDict):
    samples: int
    sigma_y: float
    sigma_1: float
    sigma_2: float
    sigma_3: float
    A: float
    B: float
    C: float


@dataclass(frozen=True)
class OrientationResult:
    index: int
    angle_radians: float
    kernel: NDArray[np.float64]
    response: NDArray[np.float64]


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


def generate_grid_from_angle(
    theta: float, samples: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate sampling coordinates while keeping the output grid fixed."""
    xs, ys = np.meshgrid(
        np.linspace(*X_RANGE, samples),
        np.linspace(*Y_RANGE, samples),
    )
    return (
        xs * np.cos(theta) + ys * np.sin(theta),
        -xs * np.sin(theta) + ys * np.cos(theta),
    )


def generate_kernel(
    theta: float = 0.0,
    samples: int = 101,
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
    if samples < 3 or samples % 2 == 0:
        raise ValueError("The kernel needs an odd sample count of at least three.")
    xs, ys = generate_grid_from_angle(theta, samples)
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


def convolve_image_with_all_orientations(
    image: NDArray[np.float64], params: KernelParameters
) -> list[OrientationResult]:
    """Return one kernel and response map for each tangent orientation."""
    results = []
    for index in range(ORIENTATION_COUNT):
        theta = index * np.pi / ORIENTATION_COUNT
        kernel = generate_kernel(theta=theta, **params)
        response = convolve_image_with_kernel(kernel, image)
        results.append(OrientationResult(index, theta, kernel, response))

    test = results[0].response
    print('orientation 0 resp=',test.min(), test.max())
    return results


def softmax_results(
    results: list[OrientationResult],
    temperature: float = 1.0,
) -> NDArray[np.float64]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    responses = np.stack([r.response for r in results], axis=0)
    # Shape: (16, height, width)
    return softmax(responses / temperature, axis=0)


def load_greyscale_img(path: str | PathLike[str]) -> NDArray[np.float64]:
    """Read an image as grayscale values between zero and one."""
    with Image.open(path) as source:
        return np.asarray(source.convert("L"), dtype=np.float64) / 255.0


def save_greyscale_img(image: NDArray[np.float64], path: str | PathLike[str]) -> None:
    """Save grayscale values as an 8-bit PNG."""
    pixels = np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def main() -> None:
    curves_dir = Path("curves")
    image_path = curves_dir / "fingerprint.png"
    image = load_greyscale_img(image_path)
    kernel_params = KernelParameters(
        sigma_y=1.0,
        sigma_1=  0.8,
        sigma_2=  0.3,
        sigma_3= 0.8,
        A= 0.5,
        B=1.5,
        C=0.5,
        samples=101,
    )
    or_res = convolve_image_with_all_orientations(image, kernel_params)
    softmax_results(or_res)


if __name__ == "__main__":
    main()
