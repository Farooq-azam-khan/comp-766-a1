"""Launch the interactive Gaussian convolution UI.

Run: python src/main.py [path/to/image.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from convolution_ui import ConvolutionUI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", type=Path, default=Path(__file__).parent / "curves/spaghetti.png"
    )
    args = parser.parse_args()
    ui = ConvolutionUI(args.image)  # Retain widget callbacks while the window is open.
    plt.show()
    ui.timer.stop()


if __name__ == "__main__":
    main()
