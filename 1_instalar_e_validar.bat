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
set LOG=logs\1_instalar_e_validar.txt
echo INICIO %date% %time% > "%LOG%"
%PY% -m pip install --disable-pip-version-check -q -e ".[dev]" >> "%LOG%" 2>&1
echo === testes === >> "%LOG%"
%PY% -m pytest -q >> "%LOG%" 2>&1
echo PYTEST_EXIT=%errorlevel% >> "%LOG%"
echo === notebook 00 === >> "%LOG%"
%PY% -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "notebooks\00_validacao_simulada.ipynb" >> "%LOG%" 2>&1
echo NB00_EXIT=%errorlevel% >> "%LOG%"
%PY% -m jupyter nbconvert --to html "notebooks\00_validacao_simulada.ipynb" >> "%LOG%" 2>&1
echo FIM %date% %time% >> "%LOG%"
