---
description: codebase
---

You will mainly work within two workspaces.

The current workspace is go2_lidar.

When tackling these topics, no need to use the rag tools, you can rely on the codbase.

1. go2_lidar workspace
- Purpose: training a quadruped robot (Unitree go2) locomotion policy for rough terrains.
- Path: /mnt/D/dev/robotics/nvidia/isaaclab/go2_lidar
- Core files located at /mnt/D/dev/robotics/nvidia/isaaclab/go2_lidar/source/go2_lidar/go2_lidar/tasks/direct/go2_lidar (env, agents, cfg, ...):
=> The environment: Isaac-Velocity-Rough-Go2-CNN-RNN-Seq-Lidar-Direct-v0 (path: /mnt/D/dev/robotics/nvidia/isaaclab/go2_lidar/source/go2_lidar/go2_lidar/tasks/direct/go2_lidar)
=> The Agent: Go2LidarRoughCNNRNNSeqPPORunnerCfg (path: /mnt/D/dev/robotics/nvidia/isaaclab/go2_lidar/source/go2_lidar/go2_lidar/tasks/direct/go2_lidar/agents/rsl_rl_ppo_cfg.py)
=> The Config: Go2LidarRoughEnvCfg (path: /mnt/D/dev/robotics/nvidia/isaaclab/go2_lidar/source/go2_lidar/go2_lidar/tasks/direct/go2_lidar/go2_lidar_env_cfg.py)
=> The Environment: Go2LidarCNNEnv (path: /mnt/D/dev/robotics/nvidia/isaaclab/go2_lidar/source/go2_lidar/go2_lidar/tasks/direct/go2_lidar/go2_cnn_lidar_env.py)

2. Huro workspace
- Purpose: Deploy the policy trained in go2_lidar workspace.
- Path: /mnt/D/dev/robotics/ros2/huro.
- Core files located at /mnt/D/dev/robotics/ros2/huro/apps/rl_tasks/locomotion/go2

3. IsaacLab workspace:
- Purpose: Retrieve informations about isaaclab's classes. This is the source code cloned from the official GitHub repository.
- Path: /mnt/D/dev/robotics/nvidia/isaaclab/isaaclab_classic

The goal is to overcome the sim to real gap.
To do so, it is required to have no mismatch between the training and the deployement configurations.
Have already been checked:
- action scale
- joint order
- frequency
- observation scale
- action smoothing is a parameter that is not the cause of the potential problems encoutered at depoyement
- the model is loaded properly.

