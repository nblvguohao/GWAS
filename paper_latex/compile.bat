@echo off
echo ==========================================
echo PlantHGNN Paper Compilation Script
echo ==========================================
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: pdflatex not found in PATH
    echo Please install TeX Live or MiKTeX
    exit /b 1
)

echo Compiling main paper...
cd main

echo - First pass...
pdflatex -interaction=nonstopmode main.tex > compile_main.log 2>&1

echo - Running bibtex...
bibtex main >> compile_main.log 2>&1

echo - Second pass...
pdflatex -interaction=nonstopmode main.tex >> compile_main.log 2>&1

echo - Final pass...
pdflatex -interaction=nonstopmode main.tex >> compile_main.log 2>&1

if exist main.pdf (
    echo.
    echo [OK] Main paper compiled successfully!
    echo Output: main\main.pdf
) else (
    echo.
    echo [ERROR] Main paper compilation failed
    echo Check main\compile_main.log for details
)

cd ..
echo.
echo Compiling supplementary material...
cd supplementary

echo - First pass...
pdflatex -interaction=nonstopmode supplementary.tex > compile_supp.log 2>&1

echo - Second pass...
pdflatex -interaction=nonstopmode supplementary.tex >> compile_supp.log 2>&1

if exist supplementary.pdf (
    echo.
    echo [OK] Supplementary material compiled successfully!
    echo Output: supplementary\supplementary.pdf
) else (
    echo.
    echo [ERROR] Supplementary compilation failed
    echo Check supplementary\compile_supp.log for details
)

cd ..
echo.
echo ==========================================
echo Compilation Complete!
echo ==========================================
echo.
echo Output files:
echo - Main paper: main\main.pdf
echo - Supplementary: supplementary\supplementary.pdf
echo.
pause
