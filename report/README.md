# Build and edit the report

`main.tex` contains the completed report, with methods, parameter choices, measured results, assumptions, limitations, and an AI-use statement. Images, table rows, and plot legends are ordered spaghetti, fingerprint, then hair. Add your name and student ID in the `\author{}` field. Keep the report within ten pages after editing.

Build from the project root:

```sh
make report
```

This runs `pdflatex` twice to resolve references and writes `report/main.pdf`. It uses the existing figures. Both `make` and `pdflatex` are available on this machine.

To build directly from this folder:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.pdf` is the compiled report. Rebuild after editing the source.

Regenerate all figures and measurements from the project root:

```sh
make report-assets
```

Run `make report` after regenerating the figures to update the PDF.

The generator reads `src/curves/` at original resolution. Edit its constants to tune the models, then update the parameter table and captions in `main.tex`. The shared baseline uses 31 kernel samples, temperature 0.005, radius 5, seven curvature classes, and five updates. Preview UIs have their own defaults.

Generated files in `assets/` include:

- `kernel_surface.png` and `kernels_all.png`: reference filter, transverse profile, and all 16 kernels.
- `<image>_original.png`, `<image>_responses.png`, and `<image>_initial.png`: inputs, 16 response contact sheets, and initial overlays.
- `<image>_initial_p05.png`: the same initial probabilities displayed at $p>0.5$ for comparison with the $p>0.2$ overlays.
- `<image>_support.png`: supported candidates and raw support heatmap.
- `<image>_relaxation.png` and `<image>_detail.png`: initial versus final estimates and central crops of all three stages.
- `support_history.png`, `support_history.csv`, and `results_table.tex`: plots, numeric histories, and the report table.
- `synthetic_crossing.png` and its metrics: a controlled crossing and isolated tangent with a separate recorded configuration.
- `manifest.json` and `<image>_metrics.json`: parameters, thresholds, crop coordinates, counts, and scores.

The report discusses central crops, the real spaghetti crossing, and the controlled crossing test. It does not claim verified fingerprint bifurcation recovery. Extra response contact sheets and original images are available for further analysis. Thresholds differ across initial, support, and final views; these differences are explained in the report.
