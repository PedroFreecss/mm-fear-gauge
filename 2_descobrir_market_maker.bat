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
set LOG=logs\2_descoberta.txt
echo Procurando market makers por 30 minutos. Pode deixar esta janela aberta.
%PY% -m fear_gauge.discover --minutos 30 > "%LOG%" 2>&1
echo DESCOBERTA_EXIT=%errorlevel% >> "%LOG%"
type "%LOG%"
