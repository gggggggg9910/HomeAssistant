@echo off
REM Simple git commit and push batch script for Windows
REM This script calls the PowerShell version for better Unicode support

powershell -ExecutionPolicy Bypass -File "%~dp0commit_git.ps1"
