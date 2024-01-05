#!/bin/bash

# Set environment variables with your Git username and access token (or password)
export GIT_USERNAME="OmarFaig"
export GIT_TOKEN="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"

# Perform Git operations
git status
git add .
git commit -m "daily updates"

# Use the environment variables for Git push
git push

