# PlantHGNN Paper LaTeX Files

This directory contains the LaTeX source files for the PlantHGNN HPO paper.

## Directory Structure

```
paper_latex/
├── README.md                      # This file
├── main/
│   └── main.tex                   # Main paper
├── supplementary/
│   └── supplementary.tex          # Supplementary material
└── figures/                       # Symlink to figures directory
```

## Files Description

### Main Paper (`main/main.tex`)
- **Title**: PlantHGNN: Hyperparameter Optimization for Plant Genome Prediction via Heterogeneous Graph Neural Networks
- **Sections**:
  1. Introduction
  2. Materials and Methods
  3. Results
  4. Discussion
  5. Conclusion
  6. Data Availability
  7. Acknowledgments
  8. References

### Supplementary Material (`supplementary/supplementary.tex`)
- **Sections**:
  1. Complete HPO Experimental Results
  2. 5-Fold Cross Validation Details
  3. Model Architecture Details
  4. Dataset Details
  5. Computational Resources
  6. Code and Data Availability
  7. Additional Figures

## Compilation Instructions

### Prerequisites
Required LaTeX packages:
- `inputenc`, `fontenc`
- `amsmath`, `amssymb`
- `graphicx`
- `booktabs`, `longtable`, `multirow`, `array`
- `float`
- `hyperref`
- `natbib` or `apalike`
- `geometry`
- `xcolor`
- `algorithm`, `algpseudocode`
- `setspace`
- `lineno`
- `listings`
- `caption`, `subcaption`

### Compilation Commands

#### Using pdflatex (recommended):

```bash
# Compile main paper
cd main
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Compile supplementary
cd ../supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex
```

#### Using latexmk (automatic compilation):

```bash
# Compile main paper
cd main
latexmk -pdf main.tex

# Compile supplementary
cd ../supplementary
latexmk -pdf supplementary.tex
```

#### Using Overleaf:
1. Upload the entire `paper_latex` directory
2. Set `main/main.tex` as the main document
3. Upload figures from `submission_package/figures/`
4. Compile

### Expected Output Files

After compilation, you should have:

```
main/
├── main.pdf              # Main paper (~8-10 pages)
├── main.aux
├── main.bbl
├── main.blg
└── main.log

supplementary/
├── supplementary.pdf     # Supplementary material (~10-15 pages)
├── supplementary.aux
└── supplementary.log
```

## Figure Requirements

Place the following figure files in the `figures/` directory:
- `figure1_hpo_comparison.pdf`
- `figure2_training_curve.pdf`
- `figure3_performance_gap.pdf`
- `figure4_network_contribution.pdf`

Or create symlinks:
```bash
cd figures
ln -s ../../submission_package/figures/*.pdf .
```

## Notes

1. **References**: The main paper uses a manual bibliography. For production, consider using BibTeX/BibLaTeX with a .bib file.

2. **Figures**: Figures are referenced from the parent directory. Ensure the relative paths are correct for your setup.

3. **Line Numbers**: The supplementary material includes line numbers for review. Remove `\usepackage{lineno}` and `\linenumbers` for final submission.

4. **Page Numbers**: Check journal guidelines for page number formatting.

## Citation Format

This template uses author-year citation format (e.g., \citep{meuwissen2001prediction}).

To switch to numbered citations:
1. Replace `\usepackage{natbib}` with `\usepackage[numbers]{natbib}`
2. Replace `\bibliographystyle{apalike}` with `\bibliographystyle{plain}`
3. Update citation commands from `\citep{}` to `\cite{}`

## Journal-Specific Modifications

### For Plant Biotechnology Journal:
- Use `\documentclass[12pt,a4paper]{article}`
- Add `\usepackage{pbtemplate}` if available
- Follow author guidelines for figure/table formatting

### For Briefings in Bioinformatics:
- Use `\documentclass[biblatex]{bioinfo}`
- Follow their specific template

## Troubleshooting

### Missing Fonts
If you encounter font errors, install the following packages:
```bash
tlmgr install collection-fontsrecommended
tlmgr install ec
```

### Missing Packages
Install missing packages using:
```bash
tlmgr install <package-name>
```

Or use the full TeX Live distribution.

### Figure Not Found
Ensure figure paths are correct. Modify the `\includegraphics` path in the .tex files if needed.

## Contact

For questions about these LaTeX files, please refer to:
- GitHub: https://github.com/nblvguohao/GWAS
- Author: Lyu (安徽农业大学 AI学院)

---

**Last Updated**: 2026-04-07
**Version**: 1.0
