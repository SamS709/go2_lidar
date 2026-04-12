# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import CommandManager, CurriculumManager
from isaaclab.utils.math import quat_conjugate, quat_apply, quat_mul, quat_inv

from .go2_lidar_env_cfg import Go2LidarFlatEnvCfg, Go2LidarRoughEnvCfg

""" windows:
python .\scripts\rsl_rl\train.py --task Isaac-Velocity-Rough-Go2-Lidar-Direct-v0 --num_envs 2048 --headless
"""

class Go2LidarEnv(DirectRLEnv):
    cfg: Go2LidarFlatEnvCfg | Go2LidarRoughEnvCfg

    def __init__(self, cfg: Go2LidarFlatEnvCfg | Go2LidarRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self.command_manager = CommandManager(self.cfg.commands, self)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "def_pos"
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*_thigh")
        self._calf_ids, _ = self._contact_sensor.find_bodies(".*_calf")
        self._undesired_contact_body_ids = self._thigh_ids
        self._body_contact_info_teacher = self._thigh_ids + self._calf_ids
        self._finite_warn_counter = 0
        print("UNDESIRED CONTACTS: Thighs and Calfs")
        print("IDS: ", self._undesired_contact_body_ids)

    def _sanitize_tensor(self, tensor: torch.Tensor, name: str, clamp_abs: float | None = None) -> torch.Tensor:
        """Replace non-finite values and optionally clamp to avoid destabilizing PPO updates."""
        if not torch.isfinite(tensor).all():
            self._finite_warn_counter += 1
            # Print occasionally to avoid flooding logs while still surfacing instability.
            if self._finite_warn_counter <= 5 or self._finite_warn_counter % 500 == 0:
                print(f"[WARN] Non-finite values detected in {name}. Applying nan_to_num safeguard.")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_abs is not None:
            tensor = torch.clamp(tensor, min=-clamp_abs, max=clamp_abs)
        return tensor

    def create_gaussian_heightmap(self, h, w):
        y = torch.arange(h, device=self.device, dtype=torch.float32)
        x = torch.arange(w, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Center of the grid
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        
        # Compute 2D Gaussian
        gaussian_dist = torch.exp(((xx - cx)**2 + (yy - cy)**2) / (2 * self.cfg.sigma**2))
        
        # Normalize to create probability distribution
        gaussian_prob = gaussian_dist / gaussian_dist.sum()
        self.gaussian_prob_heightmap  = gaussian_prob.flatten()
        

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, Go2LidarRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            # self._height_scanner_critic = RayCaster(self.cfg.height_scanner_critic)
            self.scene.sensors["height_scanner"] = self._height_scanner
            # self.scene.sensors["height_scanner_critic"] = self._height_scanner_critic
            self.create_gaussian_heightmap(15, 10)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.command_manager.compute(dt=self.step_dt)
        
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        # self._robot.set_joint_position_target(self._robot.data.default_joint_pos)    
        
    def _compute_height_data(self):
        # Get sensor/robot pose in world frame
        pos_w = self._height_scanner.data.pos_w          # (N, 3)
        quat_w = self._height_scanner.data.quat_w        # (N, 4) — w, x, y, z

        # Ray hit positions in world frame
        ray_hits_w = self._height_scanner.data.ray_hits_w  # (N, H, 3)
        N, H, _ = ray_hits_w.shape

        # Transform ray hits into the robot base frame
        # 1. Translate: shift hits relative to sensor origin
        hits_relative = ray_hits_w - pos_w.unsqueeze(1)   # (N, H, 3)

        # 2. Rotate: apply inverse of robot quaternion to go from world → base frame
        quat_inv_w = quat_inv(quat_w)                      # (N, 4)
        quat_inv_w_expanded = quat_inv_w.unsqueeze(1).expand(N, H, 4)
        hits_in_base = quat_apply(
            quat_inv_w_expanded.reshape(N * H, 4),
            hits_relative.reshape(N * H, 3)
        ).reshape(N, H, 3)

        # 3. The height in the base frame is the Z component (negative = below robot)
        height_data = -hits_in_base[..., 2] - 0.5 
        return height_data      
    
    def _process_heightmap(self, height_map):
        sampled_indices = torch.multinomial(self.gaussian_prob_heightmap, self.cfg.n_zeros, replacement=True)
        height_map_actor = height_map.clone()
        height_map_actor[:, sampled_indices] = 0.0
        return height_map_actor
    
    def _get_observations(self) -> dict:
        height_data = None
        height_data_actor = None
        if isinstance(self.cfg, Go2LidarRoughEnvCfg):
            height_data = self._compute_height_data()
            height_data_actor = height_data + (2.0 * torch.rand_like(height_data) - 1.0) * float(0.01) * self.cfg.randomize
            
            height_data_actor = self._process_heightmap(height_data_actor) 
            height_data = self._sanitize_tensor(height_data, "height_data", clamp_abs=10.0)
            height_data_actor = self._sanitize_tensor(height_data_actor, "height_data_actor", clamp_abs=10.0)
            # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
            # print(height_data.reshape(self.num_envs, 15, 10).flip(1,2))            
            
            
            # x_data = (self._height_scanner.data.pos_w[:, 0].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 0]).reshape(self.num_envs, 15, 10).flip(1,2)
            # print(x_data)
            # height_data = self._get_lidar_obs()
            # print(torch.sum(height_data_actor==0.0))
        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1) > 1.0).float()
        # Extract yaw angle from quaternion and encode as sin/cos
        actor_obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_ang_vel_b + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.1) * self.cfg.randomize,
                    self._robot.data.projected_gravity_b + (2.0 * torch.rand_like(self._robot.data.projected_gravity_b) - 1.0) * float(0.05) * self.cfg.randomize,
                    self.command_manager.get_command("base_velocity"),
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01) * self.cfg.randomize,
                    self._robot.data.joint_vel + (2.0 * torch.rand_like(self._robot.data.joint_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                    height_data_actor,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        actor_obs = self._sanitize_tensor(actor_obs, "actor_obs", clamp_abs=100.0)
        
        critic_obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self.command_manager.get_command("base_velocity"),
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    foot_contacts,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        critic_obs = self._sanitize_tensor(critic_obs, "critic_obs", clamp_abs=100.0)
        observations = {"policy": actor_obs,
                        "critic": critic_obs}
        self._previous_actions = self._actions
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self.command_manager.get_command("base_velocity")[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self.command_manager.get_command("base_velocity")[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum(torch.clamp(last_air_time - 0.5, min=0.0) * first_contact, dim=1) * (
            torch.norm(self.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1
        )
        
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        # stay around default pos:        
        cmd = torch.linalg.norm(self.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        joint_deviation = torch.sum(torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1)
        is_moving = torch.logical_or(cmd > 0.0, body_vel > self.cfg.velocity_threshold)
        def_pos = torch.where(is_moving, joint_deviation, self.cfg.stand_still_scale * joint_deviation)
        
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "def_pos" : def_pos * self.cfg.def_pos_reward_scale * self.step_dt
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = self._sanitize_tensor(reward, "reward", clamp_abs=100.0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self.curriculum_manager.compute(env_ids=env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self.command_manager.reset(env_ids)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # Add x-axis offset to spawn position
        # default_root_state[:, 0] -= 3.0  # Offset in meters (change this value as needed)
        # # Rotate 45 degrees around z-axis at spawn
        # import math
        # angle = math.pi / 4  # 45 degrees
        # z_rot_quat = torch.tensor(
        #     [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)],
        #     dtype=default_root_state.dtype, device=self.device
        # ).expand(len(env_ids), -1)
        # default_root_state[:, 3:7] = quat_mul(z_rot_quat, default_root_state[:, 3:7])

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
