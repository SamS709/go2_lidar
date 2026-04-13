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


class Go2TeacherStudentEnv(Go2LidarEnv):
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

    def _build_policy_obs(self, height_data_student: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build student observation groups for CNN policy: proprio + grid."""
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        velocity_commands = self.command_manager.get_command("base_velocity")

        proprio_obs = torch.cat(
            [
                base_ang_vel
                + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.1) * self.cfg.randomize,
                projected_gravity
                + (2.0 * torch.rand_like(self._robot.data.projected_gravity_b) - 1.0) * float(0.05) * self.cfg.randomize,
                velocity_commands,
                joint_pos_rel
                + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01) * self.cfg.randomize,
                joint_vel + (2.0 * torch.rand_like(self._robot.data.joint_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                height_data_student,
                self._actions,
            ],
            dim=-1,
        )
        proprio_obs = self._sanitize_tensor(proprio_obs, "student_proprio_obs", clamp_abs=100.0)

        # Height map is 15x10 and encoded as a single channel image for CNN input.
        grid_obs = height_data_student.view(self.num_envs, 1, 15, 10)
        grid_obs = self._sanitize_tensor(grid_obs, "student_grid_obs", clamp_abs=10.0)

        return {"proprio": proprio_obs, "grid": grid_obs}

    def _get_observations(self) -> dict:
        """
        Return both student and teacher observations for distillation.
        
        Returns:
            dict with keys:
                - "policy": student observations as dict with keys:
                    - "proprio": (N, 45)
                    - "grid": (N, 1, 15, 10)
                - "teacher": teacher observations (privileged, 210 dims)
        """
        # height data (laser)
        height_data = self._compute_height_data()
        height_data_student = height_data + (2.0 * torch.rand_like(height_data) - 1.0) * float(0.01) * self.cfg.randomize
        height_data_student = self._process_heightmap(height_data_student) 
        height_data = self._sanitize_tensor(height_data, "height_data", clamp_abs=10.0)
        height_data_student = self._sanitize_tensor(height_data_student, "height_data_actor", clamp_abs=10.0)
        
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        velocity_commands = self.command_manager.get_command("base_velocity")
        
        student_obs = torch.cat(
            [
                base_ang_vel
                + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.1) * self.cfg.randomize,
                projected_gravity
                + (2.0 * torch.rand_like(self._robot.data.projected_gravity_b) - 1.0) * float(0.05) * self.cfg.randomize,
                velocity_commands,
                joint_pos_rel
                + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01) * self.cfg.randomize,
                joint_vel + (2.0 * torch.rand_like(self._robot.data.joint_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                height_data_student,
                self._actions,
            ],
            dim=-1,
        )
        
                
        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1) > 1.0).float()
        is_contact = (
            torch.max(torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._body_contact_info_teacher], dim=-1), dim=1)[0] > 1.0
        )

        teacher_obs = torch.cat(
            [
                base_ang_vel,      
                projected_gravity, 
                velocity_commands,       
                joint_pos_rel,          
                joint_vel,          
                self._actions, 
                height_data,
                base_lin_vel,                           
                foot_contacts,     
                is_contact,
            ],
            dim=-1,
        )
        
        self._previous_actions = self._actions.clone()
        
        return {
            "policy": student_obs,
            "teacher": teacher_obs,   # Teacher network sees this (210 dims)
        }


class Go2StudentEnv(Go2LidarEnv):
    """
    Go2 environment for student fine-tuning after distillation.
    
    This uses the same limited observations as the student during distillation,
    but now trains with PPO to refine the policy.
    """
    
    cfg: Go2LidarRoughEnvCfg

    def __init__(self, cfg: Go2LidarRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _get_observations(self) -> dict:
        """
        Return only student observations (same as during distillation).
        """
        height_data = self._compute_height_data()
        height_data_student = height_data + (2.0 * torch.rand_like(height_data) - 1.0) * float(0.01) * self.cfg.randomize
        height_data_student = self._process_heightmap(height_data_student) 
        height_data = self._sanitize_tensor(height_data, "height_data", clamp_abs=10.0)
        height_data_student = self._sanitize_tensor(height_data_student, "height_data_actor", clamp_abs=10.0)
        
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b 
        joint_pos_rel = (self._robot.data.joint_pos - self._robot.data.default_joint_pos) 
        joint_vel = self._robot.data.joint_vel    
        velocity_commands = self.command_manager.get_command("base_velocity")
        self._previous_actions = self._actions.clone()
        proprio_obs = torch.cat(
            [
                base_ang_vel
                + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.1) * self.cfg.randomize,
                projected_gravity
                + (2.0 * torch.rand_like(self._robot.data.projected_gravity_b) - 1.0) * float(0.05) * self.cfg.randomize,
                velocity_commands,
                joint_pos_rel
                + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01) * self.cfg.randomize,
                joint_vel + (2.0 * torch.rand_like(self._robot.data.joint_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                self._actions,
            ],
            dim=-1,
        )
        proprio_obs = self._sanitize_tensor(proprio_obs, "student_proprio_obs", clamp_abs=100.0)
        grid_obs = self._sanitize_tensor(height_data_student.view(self.num_envs, 1, 15, 10), "student_grid_obs", clamp_abs=10.0)

        return {"policy": {"proprio": proprio_obs, "grid": grid_obs}}
