# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for Go2 teacher-student distillation using Direct RL.

This demonstrates how to adapt manager-based distillation (like G1 29DOF example)
to work with Direct RL environments.
"""

from gymnasium import spaces

from isaaclab.utils import configclass

from .go2_lidar_env_cfg import Go2LidarRoughEnvCfg


@configclass
class Go2TeacherStudentEnvCfg(Go2LidarRoughEnvCfg):
    """
    Configuration for teacher-student distillation.
    """
    
    # Teacher sees student state + full clean heightmap + privileged terms.
    teacher_observation_space = 210
    
    def __post_init__(self):
        super().__post_init__()
        # Reduce number of environments for distillation training
        self.scene.num_envs = 4096
