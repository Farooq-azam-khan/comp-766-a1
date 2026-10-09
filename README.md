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

Compute support and compare the tangent overlays with:

```sh
uv run python support_ui.py
uv run python support_ui.py curves/fingerprint.png --radius 7 --classes 7
```

To reuse tuned initial estimates, add `--assignments path/to/assignments.npy`.
The assignment array must match the input image dimensions.
Adjust the initial threshold, support threshold, and vector spacing in the preview.
The supported overlay starts with the same initial candidates as the blue overlay
and keeps those that clear the support threshold. Uncheck **Initial candidates only**
to inspect every tangent hypothesis with enough raw support.
The support overlay divides raw support by its global maximum for display.
Its starting threshold selects approximately the strongest 10% of pixel maxima.
The heatmap shows raw support.
Click **Save results** to export the preview, raw support, winning curvature classes,
and parameters to `output/support/<image-name>/`. Use `--save-only` to export without
opening a window. These generated files are ignored by Git.

`generate_support()` returns support and curvature classes, each shaped
`(16, height, width)`. It initializes curvature estimates with a first pass,
then applies mutual curvature-class membership. Pass the previous class array
as `curvature_classes=` on later relaxation iterations. The default seven signed
curvature bins span `-0.2` to `0.2` inverse pixels, excluding radii below five pixels.
The straight-curvature bin uses a one-pixel transverse displacement at the
neighborhood boundary, capped at half the maximum curvature. At the default radius
of five pixels, its limits are `-0.08` and `0.08`. The remaining bins partition the
negative and positive curvature ranges. Use an odd number of classes to retain a
central straight-curvature bin.
Use `--max-curvature` to change that limit.

Run relaxation labeling with:

```sh
uv run python relaxation_labeling.py curves/fingerprint.png --iterations 5
uv run python relaxation_labeling.py curves/fingerprint.png --iterations 5 --save-only
```

Add `--assignments path/to/assignments.npy` to reuse tuned Part I estimates.
The preview shows initial tangents, relaxed tangents, and average local support
at iteration zero and after each update. `--threshold` controls the final overlay,
`--initial-threshold` controls the initial overlay, and `--spacing` sets vector spacing.
Results save to `output/relaxation/<image-name>/`, including the preview, initial
and final assignments, final raw support, curvature classes, and a manifest with
parameters, support scores, and maximum confidence changes.

`relax_labels()` reuses the Part II support calculation and carries curvature
classes between iterations. Each orientation competes with its own no-line label
using Appendix A's radial update, so multiple orientations can survive a crossing.
Final confidences stay in `[0, 1]`; they are not renormalized across orientations.
`average_local_support()` computes the assignment's sum `A(p) = sum(p * s)`.
The plot divides this sum by the number of pixels.

The signed support is `(s - support_min) / (support_max - support_min)`.
By default, `support_max = 2 * radius`, the support of a unit-confidence horizontal
line through the neighborhood, and `support_min = 0.25 * support_max`, following
Eq. 6.15 with a minimum line confidence of 0.5. This is a simple reference scale;
it does not compensate for orientation-dependent pixel counts. Set `--support-min`
and `--support-max` to tune it. Support below the minimum suppresses a tangent.
`--step-size` controls the update strength and `--tolerance` stops updates when the
maximum confidence change is small. The curvature-class maximum and changing
classes mean the support score need not increase at every iteration.
The existing default convolution parameters give weak ridge estimates on the
fingerprint image. Reuse tuned Part I assignments for ridge detection; relaxation
can also reinforce the finger boundary if it has the strongest initial estimates.
