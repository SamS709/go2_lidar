# Docker and Runpod

This folder contains the Docker image used for local development and for pushing a template image to a registry such as Docker Hub or GHCR.

## Build locally

From the repository root:

```bash
cd docker
./build.sh
```

This now builds two images:

- `go2_lidar:isaacsim-base-5.1.0` for the heavy Isaac Sim and Isaac Lab layers,
- `go2_lidar:isaacsim-5.1.0` for the project-specific layers.

If the cloud logs repository needs authentication, export the token before building:

```bash
export CLOUD_LOGS_GITHUB_TOKEN="<your-token>"
cd docker
./build.sh
```

If you want to build manually instead of using the script, the equivalent commands are:

```bash
cd docker
docker build -f Dockerfile.base -t go2_lidar:isaacsim-base-5.1.0 ..
docker build -f Dockerfile -t go2_lidar:isaacsim-5.1.0 --build-arg BASE_IMAGE=go2_lidar:isaacsim-base-5.1.0 ..
```

## Run locally

After the image is built:

```bash
cd docker
./run.sh
```

The container starts with the project mounted at `/workspace/go2_lidar`.

## Push the image for Runpod

Runpod can only use an image that is available from a registry. The local image name `go2_lidar:isaacsim-5.1.0` must be tagged and pushed first. To speed up future pushes, it is best to push the base image once as well.

### Option 1: Docker Hub

```bash
docker tag go2_lidar:isaacsim-base-5.1.0 <dockerhub-user>/go2_lidar-base:isaacsim-5.1.0
docker push <dockerhub-user>/go2_lidar-base:isaacsim-5.1.0
docker login
docker tag go2_lidar:isaacsim-5.1.0 <dockerhub-user>/go2_lidar:isaacsim-5.1.0
docker push <dockerhub-user>/go2_lidar:isaacsim-5.1.0
```

### Option 2: GitHub Container Registry

```bash
docker tag go2_lidar:isaacsim-base-5.1.0 ghcr.io/<github-user-or-org>/go2_lidar-base:isaacsim-5.1.0
docker push ghcr.io/<github-user-or-org>/go2_lidar-base:isaacsim-5.1.0
docker login ghcr.io
docker tag go2_lidar:isaacsim-5.1.0 ghcr.io/<github-user-or-org>/go2_lidar:isaacsim-5.1.0
docker push ghcr.io/<github-user-or-org>/go2_lidar:isaacsim-5.1.0
```

If you want to build and push in one step, you can also use `docker buildx build --push` with the registry tag you want.

## Use the image in Runpod

When creating the Runpod template:

1. Set the container image to the pushed registry image, for example `ghcr.io/<github-user-or-org>/go2_lidar:isaacsim-5.1.0`.
2. Set the environment variables required by Isaac Sim:

```bash
ACCEPT_EULA=Y
PRIVACY_CONSENT=Y
```

3. If you want the cloud logs repo to be cloned during build, keep `CLOUD_LOGS_GITHUB_TOKEN` available at build time.
4. Start the pod and open a shell to verify the image launches correctly.
5. If you publish the base image too, keep the same base tag available in the same registry so Docker can reuse the large Isaac Sim layers.

## Notes

- The image is currently defined in [docker-compose.yml](docker-compose.yml) as `go2_lidar:isaacsim-5.1.0`.
- If you change the image tag, update the tag in [docker-compose.yml](docker-compose.yml) before rebuilding.
- Runpod will not build this Dockerfile for you. It pulls the already-built registry image.