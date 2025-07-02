@echo off
title CyberShield Dashboard
color 0B

echo ========================================================================
echo                   CYBERSHIELD DASHBOARD
echo ========================================================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Check if Streamlit is installed
python -c "import streamlit" > nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit is not installed. Attempting to install...
    pip install streamlit
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Streamlit.
        echo Please run: pip install streamlit
        echo.
        pause
        exit /b 1
    )
)

REM Check for other required packages
echo Checking required packages...
python -c "import pandas, numpy, plotly, joblib, sklearn, folium" > nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install pandas numpy plotly joblib scikit-learn folium streamlit-folium
    if %errorlevel% neq 0 (
        echo WARNING: Some packages may not have installed correctly.
        echo The application may not work properly.
        echo.
        timeout /t 5
    )
)

echo.
echo ========================================================================
echo Starting CyberShield Dashboard...
echo.
echo Your web browser will open automatically.
echo.
echo To stop the application, close this window or press CTRL+C.
echo ========================================================================
echo.

REM Run the Streamlit app
cd /d "%~dp0"
streamlit run cybershield_dashboard.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to start the application.
    echo Please make sure all dependencies are installed.
    echo.
    pause
    exit /b 1
)

pause