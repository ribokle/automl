@echo off
REM Thin shim — invokes setup.ps1 with execution-policy bypass so it works
REM on a locked-down workstation without a one-off `Set-ExecutionPolicy` call.
REM All real logic lives in scripts\setup.ps1.

setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup.ps1" %*
exit /b %ERRORLEVEL%
