# Manuscript (submitted version)

This folder holds the anonymous manuscript submitted to IEEE IV 2027 (8 pages), as delivered by the authors, with the
repository changes listed at the end.

- `main.tex`, `sections/`, `tables/`, `figures/` (vector PDFs with PNG previews) and `refs.bib`. Compile with pdfLaTeX
  and BibTeX (`latexmk -pdf main.tex`) after placing the standard IEEE conference class files `IEEEtran.cls` and
  `IEEEtran.bst` (from CTAN) in this folder.
- `compiled/VTT_final.pdf`: the submitted PDF.
- `supporting/`: the experiment results and confidence intervals quoted by the tables and Section V (`results.json`,
  `ci.json`).
- `figure_source/`: `build_figures.py` composes Figures 1, 2 and 5 from the plotted data in `source_data/` (CSV,
  described in its README) and the assets in `assets/`, and converts the text of Figures 3 and 4 to outlines. It needs
  PyMuPDF, ReportLab, Pillow, NumPy, pandas and Ghostscript.
- `OUTLINE.md`: the traced source of the numbers in the text; the figure-derived values (for example the brake-free
  slowing shares and the steering slope ratio) are traced in `figure_source/source_data/` and `FIGURE_DATA_CHECK.json`.
  `check_draft.py`: wording and identifier checks of the sources (`python3 check_draft.py .`).
- `FIGURE_DATA_CHECK.json`, `FONT_CHECK.txt`: the authors' checks of the figure data and of the PDF fonts.

## Not in this repository

- `IEEEtran.cls` and `IEEEtran.bst`: standard CTAN files.
- The bundled Liberation Sans fonts: `build_figures.py` uses the system copy.
- The three manufacturer vehicle pictures of Figure 1. They are third-party images used for illustration and are not
  redistributed as separate files (they remain embedded in `figures/vtt_teaser.*` and the compiled PDF).
  `build_figures.py` expects them as `figure_source/assets/teaser_A_subfigures/car_SMALL-BEV_hyundai_ioniq.png`,
  `car_MID-BEV_genesis_gv60.png` and `car_LARGE-BEV_ford_f_150_lightning.png` and stops at Figure 1 without them.
- The three forward-camera stills of Figure 5, taken from contributed drives of the source logs (licence plates, people
  and readable place names blurred). They remain embedded in `figures/vtt_examples_physics.*` at the printed resolution;
  `build_figures.py` expects them in `figure_source/assets/teaser_B_subfigures/` (`frame3_launch_SMALL-BEV_hyundai_ioniq.png`,
  `frame3_brake_MID-ICE_genesis_gv70.png`, `frame3_launch_LARGE-BEV_ford_f_150_lightning.png`).
- The authors' local teaser pipeline that `figure_source/source_data/README.md` refers to (scripts, case selection,
  intermediate assets). The CSV files are the provenance of the plotted values.
- Local delivery checks of the submission package.

## Changes made for the repository

- The XMP metadata packets (a document UUID and an ISO copy of the build time) were removed from
  `figures/evidence_sufficiency.pdf` and `figures/footprint_noise_oracle.pdf`; their drawings are unchanged. The PDF
  information dictionaries of the figures and of the compiled paper keep their build dates.
- The vehicle pictures in Figure 1 and the forward-camera stills in Figure 5 are not covered by the repository licence.
  The world outline in Figure 2 is Natural Earth 1:110m (public domain).
