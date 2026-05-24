#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$script_dir"

if [[ -z "${ISAACLAB_HOME:-}" ]]; then
	export ISAACLAB_HOME="/mnt/D/dev/robotics/nvidia/isaaclab/isaaclab"
fi

docker compose -f docker-compose.yml up -d --no-build --remove-orphans
docker compose -f docker-compose.yml exec go2_lidar bash