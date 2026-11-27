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
echo Gravando o market maker escolhido em data\alvo.json por 24 horas.
echo Deixe esta janela aberta e o PC ligado. Para parar antes, aperte Ctrl+C.
%PY% -m fear_gauge.recorder --horas 24
pause
