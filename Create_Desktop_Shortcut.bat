@echo off
echo Creating desktop shortcut for CyberShield...

REM Get the current directory
set "CURRENT_DIR=%~dp0"

REM Create a shortcut on the desktop
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\CreateShortcut.vbs"
echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\CyberShield.lnk" >> "%TEMP%\CreateShortcut.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\CreateShortcut.vbs"
echo oLink.TargetPath = "%CURRENT_DIR%Start_CyberShield.bat" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.WorkingDirectory = "%CURRENT_DIR%" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Description = "CyberShield - Network Threat Detection" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.IconLocation = "%SystemRoot%\System32\SHELL32.dll,44" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Save >> "%TEMP%\CreateShortcut.vbs"

cscript //nologo "%TEMP%\CreateShortcut.vbs"
del "%TEMP%\CreateShortcut.vbs"

echo.
echo Desktop shortcut created successfully!
echo You can now start CyberShield by double-clicking the shortcut on your desktop.
echo.
pause