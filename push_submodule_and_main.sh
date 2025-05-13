#!/bin/bash

SUBMODULE_PATH="iac"
SUBMODULE_COMMIT_MSG="Update MIDAS submodule content"
MAIN_COMMIT_MSG="Update submodule reference to latest commit"

echo "🌀 Entering submodule directory: $SUBMODULE_PATH"
cd $SUBMODULE_PATH || exit 1

echo "✅ Staging and committing submodule changes..."
git add .
git commit -m "$SUBMODULE_COMMIT_MSG"
git push origin main || { echo "❌ Failed to push submodule"; exit 1; }

cd .. || exit 1
echo "🔁 Back to main repo..."

echo "✅ Staging submodule reference update..."
git add $SUBMODULE_PATH
git commit -m "$MAIN_COMMIT_MSG"

# Pull remote changes first to avoid rejection, rebase for clean history
echo "📥 Pulling latest changes from origin/main..."
git pull --rebase origin main || { echo "❌ Failed to pull and rebase"; exit 1; }

echo "🚀 Pushing changes to remote..."
git push origin main || { echo "❌ Failed to push main repo"; exit 1; }

echo "🎉 Done! Both submodule and main repo have been updated."
