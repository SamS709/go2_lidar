#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$script_dir"

build_args=()
if [[ -n "${CLOUD_LOGS_GITHUB_TOKEN:-}" ]]; then
	build_args+=(--build-arg "CLOUD_LOGS_GITHUB_TOKEN=${CLOUD_LOGS_GITHUB_TOKEN}")
fi

docker compose -f docker-compose.yml build --pull "${build_args[@]}"