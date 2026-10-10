.PHONY: report report-assets zip

ZIP := comp766_a1_submission.zip
SUBMISSION := \
	report/main.pdf \
	src/convolution.py \
	src/support_generation.py \
	src/relaxation_labeling.py \
	src/requirements.txt \
	src/curves/spaghetti.png \
	src/curves/fingerprint.png \
	src/curves/hair.png

report:
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex

report-assets:
	uv run python report/generate_assets.py

# Core code, inputs, and report only. UI files, tests, and outputs are excluded.
zip:
	rm -f $(ZIP)
	zip -X $(ZIP) $(SUBMISSION)
