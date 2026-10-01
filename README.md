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

**Save view** creates a timestamped folder under
`<image-name>_orientations/` beside the input image. It saves the original and
grayscale images, the selected orientation's kernel and response, or all 16
when **All** is selected. Each orientation has display PNGs and raw NumPy
arrays. An **All** export also includes two 4 × 4 contact sheets. The manifest
records the parameters, angles, and display range. The PNG contrast setting
changes only the displayed pixels; the `.npy` responses keep their raw values.
