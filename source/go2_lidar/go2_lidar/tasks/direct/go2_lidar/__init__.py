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
    id="Isaac-Velocity-Rough-Go2-Lidar-Direct-v0",
    entry_point=f"{__name__}.go2_lidar_env:Go2LidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-Lidar-Distillation-Direct-v0",
    entry_point=f"{__name__}.go2_distillation_env:Go2TeacherStudentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_distillation_env_cfg:Go2TeacherStudentEnvCfg",
        # Default to teacher pretraining on privileged observations.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2LidarTeacherPretrainRunnerCfg",
        # Use this explicit key with --agent for actual student distillation.
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2LidarDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-Lidar-Distillation-CNN-Direct-v0",
    entry_point=f"{__name__}.go2_distillation_env:Go2TeacherStudentCNNEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_distillation_env_cfg:Go2TeacherStudentEnvCfg",
        # Default to teacher pretraining on privileged observations.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2LidarTeacherPretrainCNNRunnerCfg",
        # Use this explicit key with --agent for actual student distillation.
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_distillation_cfg:Go2LidarDistillationCNNRunnerCfg",
    },
)

