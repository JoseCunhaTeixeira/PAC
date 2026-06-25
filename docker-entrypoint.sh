#!/bin/sh
set -e

# If the bind-mounted data/input is empty (e.g. a freshly `mkdir -p`'d host
# folder for a no-clone run), seed it with the image's baked-in demo
# profiles so a first run has something to try immediately. A folder that
# already has content (your own data, or a clone's committed demo data) is
# left untouched.
if [ -z "$(ls -A data/input 2>/dev/null | grep -vx '\.gitkeep')" ]; then
    cp -r data_demo_input/. data/input/
fi

exec "$@"
