# COMP 766 assignment 1

- `src/` contains the submission code, dependencies, and input images.
- `test/` contains algorithm and export tests.
- `report/` contains the LaTeX report, compiled PDF, and generated assets.

Run the stages from the project root:

```sh
uv sync
uv run python src/main.py
uv run python src/support_ui.py src/curves/fingerprint.png --radius 5 --classes 7
uv run python src/relaxation_labeling.py src/curves/fingerprint.png --iterations 5 --save-only
```

Use `--assignments path/to/assignments.npy` in support or relaxation to reuse tuned initial estimates. See [src/README.md](src/README.md) for modules, defaults, and submission instructions.

Run the tests:

```sh
PYTHONPATH=src MPLBACKEND=Agg uv run python -m unittest discover -s test -v
```

Generate report assets:

```sh
make report-assets
```

Edit [report/main.tex](report/main.tex), then build the PDF from the project root:

```sh
make report
```

This writes `report/main.pdf`. See [report/README.md](report/README.md) for assets and regeneration details.
