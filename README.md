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

The UI's **Save PNG** button writes a convolved image and a JSON file with the
selected parameters beside the input image.
