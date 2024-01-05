#!/bin/bash

# Set environment variables with your Git username and access token (or password)
export GIT_USERNAME="OmarFaig"
export GIT_TOKEN="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"

# Set the remote URL with credentials
git remote set-url origin "https://${GIT_USERNAME}:${GIT_TOKEN}@github.com/OmarFaig/Masterarbeit.git"

# Perform Git operations
git status
git add .
git commit -m "daily updates"

# Use the environment variables for Git push
git push
