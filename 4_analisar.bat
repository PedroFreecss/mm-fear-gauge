@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if not defined PY python --version >nul 2>&1 && set PY=python
if not defined PY (
  echo Python nao encontrado
  pause
  exit /b 1
)
set LOG=logs\4_analisar.txt
echo INICIO %date% %time% > "%LOG%"
%PY% -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "notebooks\01_fear_gauge_hyperliquid.ipynb" >> "%LOG%" 2>&1
echo NB01_EXIT=%errorlevel% >> "%LOG%"
%PY% -m jupyter nbconvert --to html "notebooks\01_fear_gauge_hyperliquid.ipynb" >> "%LOG%" 2>&1
echo FIM %date% %time% >> "%LOG%"
