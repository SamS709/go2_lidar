# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2 teacher-student distillation environment for Direct RL.

This implements the same teacher-student distillation concept as the G1 29DOF example,
but adapted for Direct RL environments instead of manager-based.
"""

from __future__ import annotations

import torch

from .go2_lidar_env import Go2LidarEnv
from .go2_lidar_env_cfg import Go2LidarRoughEnvCfg


class Go2LidarCNNEnv(Go2LidarEnv):
    """
    Go2 environment for teacher-student distillation.
    
    Key concepts:
    - Teacher gets PRIVILEGED observations (full state including linear velocity)
    - Student gets LIMITED observations (no linear velocity - must infer from other signals)
    - Both share the same action space
    - The distillation algorithm trains the student to mimic the teacher's policy
    """
    
    cfg: Go2LidarRoughEnvCfg

    def __init__(self, cfg: Go2LidarRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _get_observations(self) -> dict:

        # Unified observation construction for all env variants.
        # Build height/grid tensors if available; otherwise provide zeroed grids.
        x_cells = max(1, int((float(self.cfg.x_range[1]) - float(self.cfg.x_range[0])) / float(self.cfg.res)))
        y_cells = max(1, int((float(self.cfg.y_range[1]) - float(self.cfg.y_range[0])) / float(self.cfg.res)))

        height_data = self._compute_height_data_from_cloud(randomize=False)
        height_data_actor = self._compute_height_data_from_cloud(randomize=self.cfg.randomize)
        height_data = self._sanitize_tensor(height_data, "height_data", clamp_abs=10.0)
        height_data_actor = self._sanitize_tensor(height_data_actor, "height_data_actor", clamp_abs=10.0)
        height_data = height_data.view(self.num_envs, x_cells, y_cells).unsqueeze(1)
        height_data_actor = height_data_actor.view(self.num_envs, x_cells, y_cells).unsqueeze(1)
        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        # print(self._rots)
        # print(self._offsets)
        # print(self.reset_zeros_freq)
        # print("Height Data Sample (Actor): ", height_data_actor)
        

        # Actor (student) proprio observations — limited/noisy proprio inputs used by policy.
        actor_proprio = torch.cat(
            [
                self._robot.data.root_ang_vel_b
                + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.1) * self.cfg.randomize,
                self._robot.data.projected_gravity_b
                + (2.0 * torch.rand_like(self._robot.data.projected_gravity_b) - 1.0) * float(0.05) * self.cfg.randomize,
                self.command_manager.get_command("base_velocity"),
                self._robot.data.joint_pos
                - self._robot.data.default_joint_pos
                + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01) * self.cfg.randomize,
                self._robot.data.joint_vel
                + (2.0 * torch.rand_like(self._robot.data.joint_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                self._actions,
            ],
            dim=-1,
        )
        actor_proprio = self._sanitize_tensor(actor_proprio, "actor_proprio", clamp_abs=100.0)

        # Critic (privileged) proprio observations — include privileged sensors like base linear velocity,
        # contact flags and other privileged terms useful for value estimation.
        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1) > 1.0).float()
        is_contact = (
            torch.max(torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._body_contact_info_teacher], dim=-1), dim=1)[0] > 1.0
        )

        critic_proprio = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self.command_manager.get_command("base_velocity"),
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                foot_contacts,
                self._actions,
            ],
            dim=-1,
        )
        critic_proprio = self._sanitize_tensor(critic_proprio, "critic_proprio", clamp_abs=100.0)

        # Update previous actions and return unified observation dict (flat keys for runner grouping).
        self._previous_actions = self._actions.clone()

        return {
            "actor_proprio": actor_proprio,
            "actor_grid": height_data_actor,
            "critic_proprio": critic_proprio,
            "critic_grid": height_data,
        }
