ARG BASE_IMAGE=go2_lidar:isaacsim-base-5.1.0
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-lc"]
ARG CLOUD_LOGS_GITHUB_TOKEN=""
ARG GO2_LIDAR_REPO="https://github.com/SamS709/go2_lidar.git"
ARG GO2_LIDAR_REF="main"

RUN cd /workspace \
    && git clone "${GO2_LIDAR_REPO}" go2_lidar \
    && cd go2_lidar \
    && git checkout "${GO2_LIDAR_REF}"

RUN cd /workspace/go2_lidar \
    && python -m pip install -e source/go2_lidar

RUN cd /workspace/go2_lidar \
    && mkdir -p logs/rsl_rl \
    && cd logs/rsl_rl \
    && if [[ -n "${CLOUD_LOGS_GITHUB_TOKEN}" ]]; then \
        git clone "https://x-access-token:${CLOUD_LOGS_GITHUB_TOKEN}@github.com/SamS709/cloud_logs_isaaclab.git" .; \
    else \
        git clone https://github.com/SamS709/cloud_logs_isaaclab.git .; \
    fi

WORKDIR /workspace/go2_lidar

# ← These two lines are what RunPod needs
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]