@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "MODE=both"
set "RUNS=20"
set "SEED=7"
set "OUTPUT_DIR=publication_outputs"
set "CACHE_DIR=_cache"
set "RUN_TESTS=1"

:parse_args
if "%~1"=="" goto args_done

if /I "%~1"=="--help" goto usage
if /I "%~1"=="-h" goto usage

if /I "%~1"=="--mode" (
    if "%~2"=="" goto missing_value
    set "MODE=%~2"
    shift
    shift
    goto parse_args
)

if /I "%~1"=="--runs" (
    if "%~2"=="" goto missing_value
    set "RUNS=%~2"
    shift
    shift
    goto parse_args
)

if /I "%~1"=="--seed" (
    if "%~2"=="" goto missing_value
    set "SEED=%~2"
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

if /I "%~1"=="--cache-dir" (
    if "%~2"=="" goto missing_value
    set "CACHE_DIR=%~2"
    shift
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
if /I not "%MODE%"=="both" if /I not "%MODE%"=="real" if /I not "%MODE%"=="self-contained" (
    echo Invalid mode: %MODE%
    echo Expected one of: both, real, self-contained
    exit /b 2
)

echo.
echo RAIL publication artifact runner
echo --------------------------------
echo Mode:       %MODE%
echo Runs:       %RUNS%
echo Seed:       %SEED%
echo Output dir: %OUTPUT_DIR%
echo Cache dir:  %CACHE_DIR%
echo.
echo The full "both" mode may download external datasets and can take a while.
echo.

where uv >nul 2>nul
if "%ERRORLEVEL%"=="0" goto run_with_uv

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if not "%ERRORLEVEL%"=="0" (
        echo Could not find uv, .venv\Scripts\python.exe, or python on PATH.
        echo Install uv from https://docs.astral.sh/uv/ or create a Python virtual environment.
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
echo Installing required experiment and dev dependencies...
"%PYTHON_EXE%" -m pip install -e ".[experiments,dev]"
if errorlevel 1 goto fail

if "%RUN_TESTS%"=="1" (
    echo.
    echo Running tests...
    "%PYTHON_EXE%" -m pytest
    if errorlevel 1 goto fail
)

echo.
echo Generating publication outputs...
"%PYTHON_EXE%" experiments\run_publication.py --mode "%MODE%" --runs "%RUNS%" --seed "%SEED%" --output-dir "%OUTPUT_DIR%" --cache-dir "%CACHE_DIR%"
if errorlevel 1 goto fail
goto success

:run_with_uv
echo Using uv.
echo Syncing required experiment and dev dependencies...
uv sync --extra experiments --extra dev
if errorlevel 1 goto fail

if "%RUN_TESTS%"=="1" (
    echo.
    echo Running tests...
    uv run --extra experiments --extra dev pytest
    if errorlevel 1 goto fail
)

echo.
echo Generating publication outputs...
uv run --extra experiments --extra dev python experiments\run_publication.py --mode "%MODE%" --runs "%RUNS%" --seed "%SEED%" --output-dir "%OUTPUT_DIR%" --cache-dir "%CACHE_DIR%"
if errorlevel 1 goto fail
goto success

:success
echo.
echo Done.
echo Results, journal-ready tables, 600 DPI/vector figures, and run_manifest.json are in:
echo %CD%\%OUTPUT_DIR%
echo See JOURNAL_ARTIFACTS.md in that directory for the output index.
exit /b 0

:fail
echo.
echo Failed. Check the console output above. If run_manifest.json exists, it may contain the failing component and error.
exit /b 1

:usage
echo Usage:
echo   run_all_results.bat [--mode both^|real^|self-contained] [--runs N] [--seed N] [--output-dir DIR] [--cache-dir DIR] [--skip-tests]
echo.
echo Defaults:
echo   --mode both --runs 20 --seed 7 --output-dir publication_outputs --cache-dir _cache
echo.
echo Examples:
echo   run_all_results.bat
echo   run_all_results.bat --mode self-contained --runs 5
echo   run_all_results.bat --mode real --skip-tests
exit /b 0
