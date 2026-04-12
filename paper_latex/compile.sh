#!/bin/bash

echo "=========================================="
echo "PlantHGNN Paper Compilation Script"
echo "=========================================="
echo ""

# Check for pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found"
    echo "Please install TeX Live or MiKTeX"
    exit 1
fi

# Compile main paper
echo "Compiling main paper..."
cd main || exit 1

echo "- First pass..."
pdflatex -interaction=nonstopmode main.tex > compile_main.log 2>&1

echo "- Running bibtex..."
bibtex main >> compile_main.log 2>&1 || true

echo "- Second pass..."
pdflatex -interaction=nonstopmode main.tex >> compile_main.log 2>&1

echo "- Final pass..."
pdflatex -interaction=nonstopmode main.tex >> compile_main.log 2>&1

if [ -f "main.pdf" ]; then
    echo ""
    echo "[OK] Main paper compiled successfully!"
    echo "Output: main/main.pdf ($(du -h main.pdf | cut -f1))"
else
    echo ""
    echo "[ERROR] Main paper compilation failed"
    echo "Check main/compile_main.log for details"
fi

cd ..

# Compile supplementary
echo ""
echo "Compiling supplementary material..."
cd supplementary || exit 1

echo "- First pass..."
pdflatex -interaction=nonstopmode supplementary.tex > compile_supp.log 2>&1

echo "- Second pass..."
pdflatex -interaction=nonstopmode supplementary.tex >> compile_supp.log 2>&1

if [ -f "supplementary.pdf" ]; then
    echo ""
    echo "[OK] Supplementary material compiled successfully!"
    echo "Output: supplementary/supplementary.pdf ($(du -h supplementary.pdf | cut -f1))"
else
    echo ""
    echo "[ERROR] Supplementary compilation failed"
    echo "Check supplementary/compile_supp.log for details"
fi

cd ..

echo ""
echo "=========================================="
echo "Compilation Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "- Main paper: main/main.pdf"
echo "- Supplementary: supplementary/supplementary.pdf"
echo ""
