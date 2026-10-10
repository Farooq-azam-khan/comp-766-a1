# Run the submission code

This folder contains the Python implementation and input images. It can be copied and run independently with Python 3.13 or later.

From this folder, install dependencies and run a stage:

```sh
python -m pip install -r requirements.txt
python main.py
python support_ui.py
python relaxation_labeling.py --save-only
```

Pass an image path as the first argument to use another input. Defaults resolve from this folder regardless of the current working directory.

- `convolution.py` implements the Gaussian template, oriented convolution, softmax initialization, and tangent geometry.
- `support_generation.py` implements cocircularity, curvature classes, and support with mutual curvature consistency.
- `relaxation_labeling.py` implements the radial update and exports assignments, support, curvature classes, a preview, and score histories.
- `convolution_ui.py` and `main.py` provide controls for tuning the filter and initial overlay.
- `support_ui.py` compares initial and supported estimates.

Convolution and support previews start with a 101-pixel kernel and temperature 0.02. Relaxation starts with a 31-pixel kernel and temperature 0.005. Use `--samples` and `--temperature` to adjust relaxation. Use `--assignments path/to/assignments.npy` to reuse tuned estimates in support or relaxation. Arrays have shape `(16, height, width)`.

Kernel sample counts are odd. Curvature class counts are positive and odd. Temperature and radius are positive. Support limits satisfy `support_min < support_max`. Loaded assignments match the image dimensions. These are caller assumptions, with no custom CLI validation layer.

Initial softmax sums to one across orientations. After relaxation, each orientation is an independent tangent versus no-line confidence in `[0, 1]`, so several directions can survive a crossing. Support is recomputed using previous curvature classes. The score is `A(p) = sum(p * s)`; previews divide it by the number of pixels.

CLI output goes under `output/` in the current working directory unless `--output` is supplied. Convolution UI exports are saved beside the input image in a timestamped orientation folder.
