from os import PathLike
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import convolve


sigma_y = 1.0
sigma_1 = 0.8
sigma_2 = 0.3
sigma_3 = 0.8

A = 0.5
B = 1.5
C = 0.5

x_min, x_max = -2.0, 2.0
y_min, y_max = -2.5, 2.5
z_min, z_max = -0.6, 0.8

epsilon = np.pi / 16
m = 16

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
    lsf = A*np.exp(-xs**2/sigma_1**2)-B*np.exp(-xs**2/sigma_2**2)+C*np.exp(-xs**2/sigma_3**2)
    return lsf*np.exp(-ys**2/sigma_y**2)

def generate_kenel():
    samples = 15 # 300, 100
    xs = np.linspace(-2, 2, samples)
    ys = np.linspace(-2.5, 2.5, samples)
    X, Y = np.meshgrid(xs, ys)
    Z = G_np(X, Y, sigma_y, sigma_1,sigma_2, sigma_3, A, B,C)
    return X, Y, Z

def visualize_convolution_surface(X, Y, Z):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-0.6, 0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("G(x, y)")
    ax.view_init(elev=20, azim=-35)

    plt.tight_layout()
    plt.show()

def load_greyscale_img(file_name: PathLike):
    # Load PNG as grayscale, scaled to [0, 1].
    image = np.asarray(
        Image.open(file_name).convert("L"),
        dtype=np.float64,
    ) / 255.0
    return image

def convolve_image_with_kernel(kernel, image, ):
    # result = convolve(image, kernel, mode='reflect') # too slow
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2

    padded = np.pad(
        image,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="symmetric",
    )

    result = fftconvolve(padded, kernel, mode="valid")
    display_result = np.clip(result, 0.0, 1.0)
    return display_result

def main():
    curves_dir = Path('./curves')
    curve_path = curves_dir / 'fingerprint.png'
    print(f"Loading Image {curve_path}...")
    image = load_greyscale_img(curve_path)
    print("Generating Kernel")
    X, Y, kernel = generate_kenel()
    # Normalize to preserve constant image brightness.
    print("Normalizing Kernel")
    kernel_sum = kernel.sum()
    if np.isclose(kernel_sum, 0.0):
        raise ValueError("Cannot normalize a kernel with a zero sum.")
    kernel /= kernel_sum

    # visualize_convolution_surface(X, Y, kernel)
    print("Convolving...")
    display_result = convolve_image_with_kernel(kernel, image)
    Image.fromarray(
        np.round(display_result * 255).astype(np.uint8)
    ).save(curves_dir / f'{curve_path.stem}_convolved.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[1].imshow(kernel, cmap="RdBu_r")
    axes[1].set_title("Kernel")
    axes[2].imshow(display_result, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Convolved")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
