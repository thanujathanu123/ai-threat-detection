# CyberShield - How to Run

This document explains how to run the CyberShield Dashboard without using the command prompt.

## Quick Start Options

### Option 1: Run Directly
1. Double-click on `Start_CyberShield.bat`
2. The dashboard will start and open in your web browser automatically

### Option 2: Create a Desktop Shortcut
1. Double-click on `Create_Desktop_Shortcut.bat`
2. A shortcut will be created on your desktop
3. Double-click the shortcut whenever you want to run the dashboard

## What Each File Does

- `Start_CyberShield.bat` - Main launcher with dependency checking
- `Create_Desktop_Shortcut.bat` - Creates a desktop shortcut for easy access
- `cybershield_dashboard.py` - The main application file

## Troubleshooting

If you encounter any issues:

1. Make sure Python is installed and in your PATH
2. Try running the following command in a command prompt:
   ```
   pip install streamlit pandas numpy plotly joblib scikit-learn folium streamlit-folium
   ```
3. If the batch files don't work, you can always run manually:
   ```
   streamlit run cybershield_dashboard.py
   ```

## System Requirements

- Python 3.7 or higher
- Required Python packages (installed automatically):
  - streamlit
  - pandas
  - numpy
  - plotly
  - joblib
  - scikit-learn
  - folium
  - streamlit-folium