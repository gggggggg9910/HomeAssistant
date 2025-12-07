# Simple git commit and push PowerShell script for Windows

Write-Host "=== Git Auto Commit Script ===" -ForegroundColor Green
Write-Host ""

# Check if we're in a git repository
try {
    $null = git rev-parse --git-dir 2>$null
} catch {
    Write-Host "❌ Error: Not a git repository" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check git status
Write-Host "📋 Checking git status..." -ForegroundColor Yellow
try {
    git status --porcelain
} catch {
    Write-Host "❌ Error: Failed to check git status" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if there are any changes to commit
$changes = git status --porcelain 2>$null
if (-not $changes) {
    Write-Host "ℹ️  No changes to commit" -ForegroundColor Blue
    Read-Host "Press Enter to exit"
    exit 0
}

# Add all changes
Write-Host ""
Write-Host "➕ Adding all changes..." -ForegroundColor Cyan
try {
    git add .
} catch {
    Write-Host "❌ Error: Failed to add changes" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Commit with default message
Write-Host ""
Write-Host "💾 Committing changes..." -ForegroundColor Magenta
try {
    git commit -m "fix"
} catch {
    Write-Host "❌ Error: Failed to commit changes" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Push to remote
Write-Host ""
Write-Host "🚀 Pushing to remote..." -ForegroundColor Yellow
try {
    git push
} catch {
    Write-Host "❌ Error: Failed to push changes" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "✅ All operations completed successfully!" -ForegroundColor Green
Write-Host "🎉 Changes have been committed and pushed!" -ForegroundColor Green
Read-Host "Press Enter to exit"
