@echo off
REM Simple git commit and push batch script for Windows

echo === Git Auto Commit Script ===
echo.

REM Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Not a git repository
    pause
    exit /b 1
)

REM Check git status
echo 📋 Checking git status...
git status --porcelain >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Failed to check git status
    pause
    exit /b 1
)

REM Check if there are any changes to commit
for /f %%i in ('git status --porcelain 2^>nul') do set HAS_CHANGES=1
if not defined HAS_CHANGES (
    echo ℹ️  No changes to commit
    pause
    exit /b 0
)

REM Add all changes
echo.
echo ➕ Adding all changes...
git add .
if %errorlevel% neq 0 (
    echo ❌ Error: Failed to add changes
    pause
    exit /b 1
)

REM Commit with default message
echo.
echo 💾 Committing changes...
git commit -m "fix"
if %errorlevel% neq 0 (
    echo ❌ Error: Failed to commit changes
    pause
    exit /b 1
)

REM Push to remote
echo.
echo 🚀 Pushing to remote...
git push
if %errorlevel% neq 0 (
    echo ❌ Error: Failed to push changes
    pause
    exit /b 1
)

echo.
echo ✅ All operations completed successfully!
echo 🎉 Changes have been committed and pushed!
pause
