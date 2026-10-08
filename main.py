"""Launch the interactive Gaussian convolution UI.

Run: python main.py [path/to/image.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from convolution_ui import ConvolutionUI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", type=Path, default=Path("curves/spaghetti.png")
    )
    args = parser.parse_args()
    image_path = args.image
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")

    ui = ConvolutionUI(image_path)  # Retain widget callbacks while the window is open.
    plt.show()
    ui.timer.stop()


if __name__ == "__main__":
    main()
