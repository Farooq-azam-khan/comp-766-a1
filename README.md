# COMP 766 assignment 1

`convolution.py` contains the Gaussian difference function, kernel generation,
image convolution, and the assignment's three-image example. Run that example
with:

```sh
uv run python convolution.py
```

`convolution_ui.py` contains the Matplotlib controls and preview. `main.py`
launches that UI with the default image or another PNG:

```sh
uv run python main.py
uv run python main.py curves/hair.png
```

The preview shows the original color image, its grayscale input, a kernel, and
the convolved image. Choose an angle from the orientation menu, or choose
**All** to see all 16 kernels and responses. **Auto update** applies slider
changes after a short pause. **Auto contrast** stretches the response display
using one shared range across all 16 orientations.

The original image has a blue overlay of initial tangent estimates. Softmax
normalizes the 16 raw responses at every pixel, and the overlay draws every
orientation whose probability is strictly greater than **Vector threshold**,
which starts at **0.20**. The overlay always considers all 16 orientations,
regardless of the orientation menu. The threshold updates immediately without
recomputing convolutions.

**Softmax temp** starts at 0.02 and ranges from 0.001 to 0.1; lower it to make
assignments more concentrated when no probabilities exceed the threshold. At
temperature 1.0, the default spaghetti responses give a maximum probability of
about 0.067, so nothing clears the 0.20 threshold. The smaller default makes
vectors visible, but temperature remains a measurement-model tuning choice.
**Vector spacing** starts at 4 pixels
to keep the overlay readable. Set it to 1 to draw at every pixel. These display
controls also update immediately. Auto contrast does not affect probabilities.
Tangents follow the rotated kernel's vertical axis in image pixels, accounting
for the different horizontal and vertical sampling ranges.

**Save view** creates a timestamped folder under
`<image-name>_orientations/` beside the input image. It saves the original and
grayscale images, the selected orientation's kernel and response, or all 16
when **All** is selected. Each orientation has display PNGs and raw NumPy
arrays. An **All** export also includes two 4 × 4 contact sheets. The manifest
records the parameters, angles, and display range. The PNG contrast setting
changes only the displayed pixels; the `.npy` responses keep their raw values.
Every export also saves `initial_tangents.png`, the full per-pixel probability
array `assignments.npy` with shape `(16, height, width)`, and the threshold,
temperature, and vector spacing in the manifest.
