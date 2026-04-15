# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass



"""
MLP:
TRAIN TEACHER:
python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-Lidar-Distillation-Direct-v0 --num_envs 6144 --headless

DISTILL STUDENT:
python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-Lidar-Distillation-Direct-v0 --agent rsl_rl_distillation_cfg_entry_point --num_envs 8192 --headless

CNN + MLP:
TRAIN TEACHER:
python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-Lidar-Distillation-CNN-Direct-v0 --num_envs 6144 --headless

DISTILL STUDENT:
python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-Lidar-Distillation-CNN-Direct-v0 --agent rsl_rl_distillation_cfg_entry_point --num_envs 8192 --headless

"""


from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
    RslRlCNNModelCfg,
)


@configclass
class Go2LidarTeacherPretrainRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Pretrain the teacher policy on privileged observations before distillation."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go2_distillation"
    run_name = "teacher"

    # rsl_rl (classic) expects actor/critic observation set names for PPO.
    # Use privileged teacher observations for both actor and critic.
    obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2LidarDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go2_distillation"
    run_name = "distillation"
    # Always load from a teacher-pretrain run in this experiment folder.
    load_run = ".*_teacher"
    load_checkpoint = "model_.*.pt"
    # rsl_rl (classic) expects student/teacher observation set names for distillation.
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}
    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )



@configclass
class Go2LidarTeacherPretrainCNNRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Pretrain the teacher policy on privileged observations before distillation."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go2_distillation_cnn"
    run_name = "teacher"

    # rsl_rl expects observation routing at runner level.
    # Provide both 1D proprio and 2D exteroceptive height scans to each model.
    obs_groups = {
        "actor": ["teacher_proprio", "teacher_height_scan"],
        "critic": ["teacher_proprio", "teacher_height_scan"],
    }

    actor = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,  # normalizes only the 1D proprio path
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )

    critic = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,  # normalizes only the 1D proprio path
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2LidarDistillationCNNRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go2_distillation_cnn"
    run_name = "distillation"
    # Always load from a teacher-pretrain run in this experiment folder.
    load_run = ".*_teacher"
    load_checkpoint = "model_.*.pt"
    # Route student/teacher sets to explicit 1D+2D groups.
    obs_groups = {
        "student": ["student_proprio", "student_height_scan"],
        "teacher": ["teacher_proprio", "teacher_height_scan"],
    }
    student = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,  # normalizes only the 1D proprio path
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"),
    )
    teacher = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,  # normalizes only the 1D proprio path
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )

