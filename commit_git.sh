#!/bin/bash
# Simple git commit and push script

echo "=== Git Auto Commit Script ==="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Check git status
echo "📋 Checking git status..."
git status --porcelain
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to check git status"
    exit 1
fi

# Check if there are any changes to commit
if [ -z "$(git status --porcelain)" ]; then
    echo "ℹ️  No changes to commit"
    exit 0
fi

# Add all changes
echo ""
echo "➕ Adding all changes..."
git add .
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to add changes"
    exit 1
fi

# Commit with default message
echo ""
echo "💾 Committing changes..."
git commit -m "fix"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to commit changes"
    exit 1
fi

# Push to remote
echo ""
echo "🚀 Pushing to remote..."
git push
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to push changes"
    exit 1
fi

echo ""
echo "✅ All operations completed successfully!"
echo "🎉 Changes have been committed and pushed!"
