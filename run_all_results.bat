@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "TIER=headline"
set "WORKERS=8"
set "OUTPUT_DIR=publication_outputs"
set "RUN_TESTS=1"
set "EXTRA="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--help" goto usage
if /I "%~1"=="-h" goto usage
if /I "%~1"=="--tier" (
    if "%~2"=="" goto missing_value
    set "TIER=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--workers" (
    if "%~2"=="" goto missing_value
    set "WORKERS=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--output-dir" (
    if "%~2"=="" goto missing_value
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--fast" (
    set "EXTRA=%EXTRA% --fast"
    shift
    goto parse_args
)
if /I "%~1"=="--slow" (
    set "EXTRA=%EXTRA% --slow"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-tests" (
    set "RUN_TESTS=0"
    shift
    goto parse_args
)
echo Unknown option: %~1
goto usage

:missing_value
echo Missing value for option: %~1
exit /b 2

:args_done
echo.
echo RAIL paper reproduction runner
echo ------------------------------
echo Tier:       %TIER%
echo Workers:    %WORKERS%
echo Output dir: %OUTPUT_DIR%
echo Extra:      %EXTRA%
echo.

where uv >nul 2>nul
if "%ERRORLEVEL%"=="0" goto run_with_uv

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" -m pip install -e ".[experiments,dev]"
if errorlevel 1 goto fail
if "%RUN_TESTS%"=="1" (
    "%PYTHON_EXE%" -m pytest
    if errorlevel 1 goto fail
)
"%PYTHON_EXE%" -m experiments.reproduce_paper --tier "%TIER%" --workers "%WORKERS%" --output-root "%OUTPUT_DIR%"%EXTRA%
if errorlevel 1 goto fail
goto success

:run_with_uv
echo Using uv.
uv sync --extra experiments --extra dev
if errorlevel 1 goto fail
if "%RUN_TESTS%"=="1" (
    uv run --extra experiments --extra dev pytest
    if errorlevel 1 goto fail
)
uv run --extra experiments --extra dev python -m experiments.reproduce_paper --tier "%TIER%" --workers "%WORKERS%" --output-root "%OUTPUT_DIR%"%EXTRA%
if errorlevel 1 goto fail
goto success

:success
echo.
echo Done. Tables, figures, stats reports, and run_manifest.json are in:
echo %CD%\%OUTPUT_DIR%
echo See RUN.md for tier definitions and the backend validation note.
exit /b 0

:fail
echo.
echo Failed. Check the console output above and any run_manifest.json.
exit /b 1

:usage
echo Usage:
echo   run_all_results.bat [--tier smoke^|medium^|headline^|robustness] [--workers N] [--output-dir DIR] [--fast^|--slow] [--skip-tests]
echo.
echo Defaults:
echo   --tier headline --workers 8 --output-dir publication_outputs
echo.
echo Examples:
echo   run_all_results.bat --tier smoke --fast
echo   run_all_results.bat --tier headline --workers 8 --slow
exit /b 0
