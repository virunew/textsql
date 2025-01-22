
@echo off
REM Activate the virtual environment if needed
REM call path\to\your\venv\Scripts\activate

REM Set the PYTHONPATH to include the src directory
set PYTHONPATH=%PYTHONPATH%;%cd%\src

REM Run the tests using pytest
python -m pytest tests/test_sql_generation.py --tb=short --disable-warnings

REM Pause to see the output
pause