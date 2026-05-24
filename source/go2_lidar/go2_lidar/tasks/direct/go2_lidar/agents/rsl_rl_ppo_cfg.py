# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab_rl.rsl_rl.rl_cfg import RslRlCNNModelCfg


@configclass
class Go2LidarFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "go2_lidar"
    # Route observations as 1D proprio + 2D grid for both actor and critic.
    obs_groups = {
        "actor": ["actor_proprio", "actor_grid"],
        "critic": ["critic_proprio", "critic_grid"],
    }

    # Use distinct CNN-based actor and critic models that fuse proprio + grid.
    actor = RslRlCNNModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.8, std_type="log"),
    )

    critic = RslRlCNNModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="avg",
        ),
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.8, std_type="log"),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
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
class Go2LidarRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go2_lidar"
    obs_groups = {
        "actor": ["actor_proprio", "actor_grid"],
        "critic": ["critic_proprio", "critic_grid"],
    }
    actor = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[5, 3],         
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="none",         
        ),
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(
            init_std=0.8, std_type="log"
        ),
    )

    critic = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[5, 3],
            stride=[2, 2],
            activation="relu",
            max_pool=False,
            global_pool="none",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
