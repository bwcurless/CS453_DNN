#!/bin/bash
# My script for pushing Assignment to monsoon
dirname="$(basename "$(pwd)")"
destname="wind:~/$dirname"
echo "Copying all files in directory $dirname to $destname"
# Copy actual file instead of symlinks
rsync -a --copy-links --exclude='.*.swp' ./ "$destname"
