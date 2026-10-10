.PHONY: report report-assets

report:
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex

report-assets:
	uv run python report/generate_assets.py
