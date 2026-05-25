#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$script_dir"

build_args=()
if [[ -n "${CLOUD_LOGS_GITHUB_TOKEN:-}" ]]; then
	build_args+=(--build-arg "CLOUD_LOGS_GITHUB_TOKEN=${CLOUD_LOGS_GITHUB_TOKEN}")
fi

base_image="go2_lidar:isaacsim-base-5.1.0"
app_image="go2_lidar:isaacsim-5.1.0"

docker build \
	-f Dockerfile.base \
	-t "$base_image" \
	"${build_args[@]}" \
	..

docker build \
	-f Dockerfile \
	-t "$app_image" \
	--build-arg "BASE_IMAGE=${base_image}" \
	"${build_args[@]}" \
	..