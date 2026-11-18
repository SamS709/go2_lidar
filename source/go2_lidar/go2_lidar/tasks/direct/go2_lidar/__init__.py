# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Go2-Lidar-Direct-v0",
    entry_point=f"{__name__}.go2_lidar_env:Go2LidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-MLP-Lidar-Direct-v0",
    entry_point=f"{__name__}.go2_lidar_env:Go2LidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarRoughMLPPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-RNN-Lidar-Direct-v0",
    entry_point=f"{__name__}.go2_lidar_env:Go2LidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarRoughRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-CNN-Lidar-v0",
    entry_point=f"{__name__}.go2_lidar_env:Go2LidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarRoughCNNPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2LidarTeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2LidarDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-CNN-Distillation-Lidar-v0",
    entry_point=f"{__name__}.go2_distillation_env:Go2TeacherStudentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarRoughEnvCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2LidarTeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2LidarDistillationRunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-Go2-CNN3D-Lidar-Direct-v0",
    entry_point=f"{__name__}.go2_cnn3d_lidar_env:Go2LidarCNN3DEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarRoughCNN3DPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-CNN-RNN-Seq-Lidar-Direct-v0",
    entry_point=f"{__name__}.go2_cnn_lidar_env:Go2LidarCNNEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarRoughCNNRNNSeqPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-CNN-RNN-Lidar-Direct-v0",
    entry_point=f"{__name__}.go2_lidar_env:Go2LidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarRoughCNNRNNPPORunnerCfg",
    },
)


