#!/usr/bin/env bash
set -euo pipefail

# Create cache dirs if they don't exist (RunPod has no pre-made host dirs)
mkdir -p /root/.cache/ov
mkdir -p /root/.cache/pip
mkdir -p /root/.cache/nvidia/GLCache
mkdir -p /root/.nv/ComputeCache
mkdir -p /root/.nvidia-omniverse/logs
mkdir -p /root/.local/share/ov/data

# If a command is passed (e.g. from RunPod's "Docker Command" field), run it
# Otherwise run the default training
if [[ $# -gt 0 ]]; then
    exec "$@"
fi

# Default: headless training
exec sleep infinity