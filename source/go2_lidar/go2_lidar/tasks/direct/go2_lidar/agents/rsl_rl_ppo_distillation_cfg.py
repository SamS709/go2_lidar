# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass



"""
TRAIN TEACHER:
python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-CNN-Distillation-Lidar-v0 --agent rsl_rl_teacher_cfg_entry_point --num_envs 3072 --headless

DISTILL STUDENT:
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-CNN-Distillation-Lidar-v0 --agent rsl_rl_distillation_cfg_entry_point --resume --load_run <teacher_run_folder> --checkpoint model_<N>.pt --num_envs 3072 --headless
    
"""


from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)
from isaaclab_rl.rsl_rl.rl_cfg import RslRlCNNModelCfg


from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import Go2LidarRoughCNNPPORunnerCfg

def _cnn():
    # must be identical to the PPO teacher's CNN so the checkpoint loads
    return RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32],
        kernel_size=[3, 3],
        stride=[1, 1],
        padding="zeros",
        activation="relu",
        max_pool=False,
        global_pool="none",
    )


@configclass
class Go2LidarDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 100000
    save_interval = 50
    experiment_name = "go2_lidar_distillation"

    obs_groups = {
        "student": ["student_proprio", "student_grid"],   # history proprio + noisy grid
        "teacher": ["teacher_proprio", "teacher_grid"],     # privileged, clean
    }

    # Student: sees only deployable obs
    student = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=_cnn(),
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"),
    )

    # Teacher: same architecture as the PPO actor you trained
    teacher = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=_cnn(),
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"),
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",     # "huber" is also available
    )


@configclass
class Go2LidarTeacherPPORunnerCfg(Go2LidarRoughCNNPPORunnerCfg):
    experiment_name = "go2_lidar_teacher"
    obs_groups = {
        "actor": ["teacher_proprio", "teacher_grid"],    # privileged, clean
        "critic": ["teacher_proprio", "teacher_grid"],
    }
