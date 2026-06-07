# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import CommandManager, CurriculumManager
from isaaclab.utils.math import quat_conjugate, quat_apply, quat_mul, quat_inv, quat_rotate_inverse

from .go2_lidar_env_cfg import Go2LidarFlatEnvCfg, Go2LidarRoughEnvCfg

""" windows:
python ./scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-Lidar-Direct-v0 --num_envs 2048 --headless
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
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self.command_manager = CommandManager(self.cfg.commands, self)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        self._desired_hip_offset = torch.tensor([-self.cfg.desired_hip_offset, self.cfg.desired_hip_offset, -self.cfg.desired_hip_offset, self.cfg.desired_hip_offset], device=self.device)
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "track_height",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "dof_energy_l2",
                "action_rate_l2",
                "action_rate_2_l2",
                "feet_air_time",
                "feet_gait",
                # "feet_dist",
                # "undesired_contacts",
                # "flat_orientation_l2",
                "def_pos",
                "feet_to_hip",
                "feet_grounded_stop",
                # "feet_vertical_surface"
            ]
        }
        # Get specific body indices
        self._base_id_sensor, _base_name = self._contact_sensor.find_bodies("base")
        self._feet_ids_sensor, _feet_name = self._contact_sensor.find_bodies(".*_foot")
        self._thigh_ids_sensor, _thigh_name = self._contact_sensor.find_bodies(".*_thigh")
        self._hip_ids_sensor, _hip_name = self._contact_sensor.find_bodies(".*_hip")
        self._calf_ids_sensor, _calf_name = self._contact_sensor.find_bodies(".*_calf")
        
        self._base_id, _ = self._robot.find_bodies("base")
        self._feet_ids, _ = self._robot.find_bodies(".*_foot")
        self._thigh_ids, _ = self._robot.find_bodies(".*_thigh")
        self._hip_ids, _ = self._robot.find_bodies(".*_hip")
        self._calf_ids, _ = self._robot.find_bodies(".*_calf")
        
        self._undesired_contact_body_ids_sensor = self._thigh_ids_sensor
        self._body_contact_info_teacher_sensor = self._base_id_sensor + self._thigh_ids_sensor + self._calf_ids_sensor
        self._finite_warn_counter = 0
        
    def build_col_to_subterrain(self):
        num_cols = self._terrain.cfg.terrain_generator.num_cols
        col_to_name = {}
        col = 0
        for name, sub_cfg in self._terrain.cfg.terrain_generator.sub_terrains.items():
            n_cols = round(sub_cfg.proportion * num_cols)
            for _ in range(n_cols):
                if col < num_cols:
                    col_to_name[col] = name
                col += 1
        return col_to_name
    
    def build_terrain_mask(self):
        for terrain_name in self._terrain.cfg.terrain_generator.sub_terrains.keys():
            col_to_bool = torch.zeros(self._terrain.cfg.terrain_generator.num_cols, dtype=torch.bool, device=self._terrain.device)
            for col, name in self.build_col_to_subterrain().items():
                col_to_bool[col] = (name == terrain_name)
            # Index by terrain_types once — this never changes
            self._terrain_masks[terrain_name] = col_to_bool[self._terrain.terrain_types]

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

    def _create_gaussian_heightmap(self, h, w):
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
        self.sampled_indices = torch.multinomial(self.gaussian_prob_heightmap, self.cfg.n_zeros, replacement=True)
        self.same_zeros_count = 0
        self.reset_zeros_freq = int(torch.randint(1, self.cfg.max_reset_zeros_freq + 1, (1,), device=self.device).item())        
    
    def _apply_yaw_rotation(self, points: torch.Tensor) -> torch.Tensor:
        angles = torch.deg2rad(self._rots).unsqueeze(-1)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        x_coord = points[..., 0]
        y_coord = points[..., 1]
        z_coord = points[..., 2]
        rotated_x = x_coord * cos_angles + z_coord * sin_angles
        rotated_z = -x_coord * sin_angles + z_coord * cos_angles
        return torch.stack((rotated_x, y_coord, rotated_z), dim=-1)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, Go2LidarRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            # Previous initialization path kept for reference:
            # self._height_scanner = RayCaster(self.cfg.height_scanner)
            self._height_scanner = self.cfg.height_scanner.class_type(self.cfg.height_scanner)
            # self._height_scanner_critic = RayCaster(self.cfg.height_scanner_critic)
            self.scene.sensors["height_scanner"] = self._height_scanner
            # self.scene.sensors["height_scanner_critic"] = self._height_scanner_critic
            x_cells = max(1, int((self.cfg.x_range[1] - self.cfg.x_range[0]) / self.cfg.res))
            y_cells = max(1, int((self.cfg.y_range[1] - self.cfg.y_range[0]) / self.cfg.res))
            self._create_gaussian_heightmap(x_cells, y_cells)
            self._rots = torch.empty(self.num_envs, device=self.device)
            self._offsets = torch.empty(self.num_envs, device=self.device)
            self._rots.uniform_(-self.cfg.max_rot, self.cfg.max_rot)
            self._offsets.uniform_(-self.cfg.max_offset, self.cfg.max_offset)
            # self._rots = torch.tensor()
            

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._subterrain_names = list(self._terrain.cfg.terrain_generator.sub_terrains.keys())
        self._terrain_masks = {}
        self.build_terrain_mask()
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
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        if self.cfg.clamp_actions:
            self._actions = torch.clamp(self._actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)
        
        if(self.cfg.filter_actions):
            alpha = 0.8
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp + self._robot.data.default_joint_pos
        else:
            self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        # self._robot.set_joint_position_target(self._robot.data.default_joint_pos)    
        
    def _compute_height_data(self, method, randomize: bool = False):
        if method == "normal":
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - self.cfg.desired_base_height
            ).clip(-1.0, 1.0) 
            if randomize and hasattr(self, "_rots"):
                ray_hits_w = self._height_scanner.data.ray_hits_w
                ray_hits_rel = ray_hits_w - self._height_scanner.data.pos_w.unsqueeze(1)
                ray_hits_rel = self._apply_yaw_rotation(ray_hits_rel)
                height_data = (self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - ray_hits_rel[..., 2] - self.cfg.desired_base_height).clip(-1.0, 1.0)
                height_data = self._apply_offset(height_data)  
                height_data += (2.0 * torch.rand_like(height_data) - 1.0) * float(0.01)
                height_data = self._zero_heightmap_cells(height_data)       
            return height_data
        else:            
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

            if randomize and hasattr(self, "_rots"):
                hits_in_base = self._apply_yaw_rotation(hits_in_base)

            # 3. The height in the base frame is the Z component (negative = below robot)
            height_data = -hits_in_base[..., 2] - self.cfg.desired_base_height
            if randomize:
                height_data = self._apply_offset(height_data)
                height_data += (2.0 * torch.rand_like(height_data) - 1.0) * float(0.01)
                height_data = self._zero_heightmap_cells(height_data) 
            return height_data      

    def _compute_height_data_from_cloud(self, randomize: bool = False):
        """Compute flattened heightmap in lidar frame using cfg x/y bounds and cell size."""
        data = self._height_scanner.data
        ray_hits_w = data.ray_hits_w
        lidar_pos_w = data.pos_w
        lidar_quat_w = data.quat_w
        num_envs, num_rays, _ = ray_hits_w.shape
        rays_rel_w = - ray_hits_w + lidar_pos_w.unsqueeze(1)
        rays_lidar = quat_apply(
            quat_conjugate(lidar_quat_w).unsqueeze(1).expand(num_envs, num_rays, 4).reshape(-1, 4),
            rays_rel_w.reshape(-1, 3),
        ).reshape(num_envs, num_rays, 3)
        # rays_lidar = rays_rel_w
        if randomize and hasattr(self, "_rots"):
            rays_lidar = self._apply_yaw_rotation(rays_lidar)

        cell_size_m = float(self.cfg.res)
        inv_cell_size = 1.0 / cell_size_m
        x_min, x_max = float(self.cfg.x_range[0]), float(self.cfg.x_range[1])
        y_min, y_max = float(self.cfg.y_range[0]), float(self.cfg.y_range[1])
        x_cells = max(1, int((x_max - x_min) / cell_size_m))
        y_cells = max(1, int((y_max - y_min) / cell_size_m))
        num_cells = x_cells * y_cells

        rays_flat = rays_lidar.reshape(-1, 3)
        env_ids = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(num_envs, num_rays).reshape(-1)

        valid = torch.isfinite(rays_flat).all(dim=1)
        if not torch.any(valid):
            return torch.zeros((num_envs, num_cells), device=self.device)

        rays_valid = rays_flat[valid]
        env_ids = env_ids[valid]

        x_idx = torch.floor((rays_valid[:, 0] - x_min) * inv_cell_size).long()
        y_idx = torch.floor((rays_valid[:, 1] - y_min) * inv_cell_size).long()
        in_bounds = (x_idx >= 0) & (x_idx < x_cells) & (y_idx >= 0) & (y_idx < y_cells)
        if not torch.any(in_bounds):
            return torch.zeros((num_envs, num_cells), device=self.device)

        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
        env_ids = env_ids[in_bounds]
        z_vals = rays_valid[in_bounds, 2]

        # Flatten env and cell indexing into one reduce op.
        flat_idx = env_ids * num_cells + x_idx * y_cells + y_idx
        height_map = torch.full((num_envs * num_cells,), -torch.inf, device=self.device)
        height_map.scatter_reduce_(0, flat_idx, z_vals, reduce="amax", include_self=True)
        height_map = torch.where(torch.isfinite(height_map), -height_map, torch.zeros_like(height_map))
        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        
        # print(height_map + self.cfg.desired_base_height)
        height_map = height_map.reshape(num_envs, num_cells) - self.cfg.desired_base_height
        if randomize:
            height_map = self._apply_offset(height_map)
            height_map += (2.0 * torch.rand_like(height_map) - 1.0) * float(0.01)
            height_map = self._zero_heightmap_cells(height_map)            
            
        # Keep ordering consistent with lidar_debug flow.
        return height_map
    

    def _apply_offset(self, height_map):
        if not hasattr(self, "_offsets"):
            return height_map
        offset_shape = (self._offsets.shape[0],) + (1,) * (height_map.ndim - 1)
        return height_map + self._offsets.view(offset_shape)
    
    def _zero_heightmap_cells(self, height_map):
        self.same_zeros_count += 1
        if self.same_zeros_count == self.reset_zeros_freq:
            self.reset_zeros_freq = int(torch.randint(1, self.cfg.max_reset_zeros_freq + 1, (1,), device=self.device).item())            
            self.same_zeros_count = 0   
            self.sampled_indices = torch.multinomial(self.gaussian_prob_heightmap, self.cfg.n_zeros, replacement=True)
        height_map_actor = height_map.clone()
        height_map_actor[:, self.sampled_indices] = 0.0
        return height_map_actor
    
    def is_on_terrain(self, terrain_names: list[str]) -> torch.Tensor:
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name in terrain_names:
            mask |= self._terrain_masks[name]
        return mask
    
    def _get_observations(self) -> dict:
        
        height_data = self._compute_height_data_from_cloud(randomize=False)
        height_data_actor = self._compute_height_data_from_cloud(randomize=self.cfg.randomize)
        height_data = self._sanitize_tensor(height_data, "height_data", clamp_abs=10.0)
        height_data_actor = self._sanitize_tensor(height_data_actor, "height_data_actor", clamp_abs=10.0)

        noise = lambda t, s: (2.0 * torch.rand_like(t) - 1.0) * s * self.cfg.randomize
        
        # x_cells = max(1, int((float(self.cfg.x_range[1]) - float(self.cfg.x_range[0])) / float(self.cfg.res)))
        # y_cells = max(1, int((float(self.cfg.y_range[1]) - float(self.cfg.y_range[0])) / float(self.cfg.res)))
        # height_data_print = height_data.view(self.num_envs, x_cells, y_cells).flip(dims=[1]).unsqueeze(1)
        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        
        # print(height_data_print + self.cfg.desired_base_height)
        
        actor_proprio = torch.cat([
            self._robot.data.root_ang_vel_b + noise(self._robot.data.root_ang_vel_b, 0.1),
            self._robot.data.projected_gravity_b + noise(self._robot.data.projected_gravity_b, 0.05),
            self.command_manager.get_command("base_velocity"),
            self._robot.data.joint_pos - self._robot.data.default_joint_pos + noise(self._robot.data.joint_pos, 0.01),
            self._robot.data.joint_vel + noise(self._robot.data.joint_vel, 0.1),
            self._actions,
        ], dim=-1)
        actor_proprio = self._sanitize_tensor(actor_proprio, "actor_proprio", clamp_abs=100.0)

        actor_grid = self._sanitize_tensor(height_data_actor, "actor_grid", clamp_abs=10.0)

        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor], dim=-1) > 1.0).float()
        
        critic_proprio = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self.command_manager.get_command("base_velocity"),
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            foot_contacts,
            self._actions,
        ], dim=-1)
        
        if self.cfg.filter_actions:
            actor_proprio = torch.cat([actor_proprio, self._previous_actions], dim=-1)
            critic_proprio = torch.cat([critic_proprio, self._previous_actions], dim=-1)
        
        critic_proprio = self._sanitize_tensor(critic_proprio, "critic_proprio", clamp_abs=100.0)

        critic_grid = self._sanitize_tensor(height_data, "critic_grid", clamp_abs=10.0)
       

        return {
            "actor_proprio": actor_proprio,
            "actor_grid":    actor_grid,
            "critic_proprio": critic_proprio,
            "critic_grid":   critic_grid,
        }

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self.command_manager.get_command("base_velocity")[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self.command_manager.get_command("base_velocity")[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # track_height
        height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
        height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5) # Handle inf values
        mean_height_ray = torch.mean(height_data_scanner, dim=1)
        height_error = torch.square(self.cfg.desired_base_height + mean_height_ray - self._robot.data.root_state_w[:, 2])
        height_error_mapped = torch.exp(-height_error / 0.01)
        
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # joint energy
        joint_energy = torch.sum(torch.abs(self._robot.data.applied_torque * self._robot.data.joint_vel), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # action rate, order 2:
        action_rate_2 = torch.sum(torch.square(self._actions - 2*self._previous_actions + self._previous_previous_actions), dim=1)        
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids_sensor]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids_sensor]
        air_time = torch.sum(torch.clamp(last_air_time - 0.5, min=0.0) * first_contact, dim=1) * (
            torch.norm(self.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1
        )
        
        # useful for next two rewards:
        cmd = torch.linalg.norm(self.command_manager.get_command("base_velocity"), dim=1)
        should_move = cmd > 0.01 

        # gait trot
        foot_contact = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor], dim=-1) > 1.0
        # FEET iDS:  [4, 8, 14, 18]
        # RL ID:  ([14], ['RL_foot'])
        # FL ID:  ([4], ['FL_foot'])
        # RR ID:  ([18], ['RR_foot'])
        # FR ID:  ([8], ['FR_foot'])
        # Pair A in air: FL(0) and RR(3) off ground simultaneously
        pair_A_air = (~foot_contact[:, 0]) & (~foot_contact[:, 3])  # [N]
        # Pair B in air: FR(1) and RL(2) off ground simultaneously
        pair_B_air = (~foot_contact[:, 1]) & (~foot_contact[:, 2])  # [N]
        # reward when either valid diagonal pair is fully airborne
        trot_pattern = (pair_A_air | pair_B_air).float()             # [N]
        # --- penalise anti-trot: wrong pairs in air simultaneously ---
        # e.g. FL+FR both up (bound) or FL+RL both up (pace) — not trot
        wrong_pair = (
            ((~foot_contact[:, 0]) & (~foot_contact[:, 1])) |  # FL+FR = bound front
            ((~foot_contact[:, 2]) & (~foot_contact[:, 3])) |  # RL+RR = bound rear
            ((~foot_contact[:, 0]) & (~foot_contact[:, 2])) |  # FL+RL = pace left
            ((~foot_contact[:, 1]) & (~foot_contact[:, 3]))    # FR+RR = pace right
        ).float()
        # --- only reward when actually moving ---
        gait = (trot_pattern - 0.5 * wrong_pair ).clamp(min=0.0) * should_move
        
        all_feet_grounded = (
            foot_contact[:, 0] &  # FL
            foot_contact[:, 1] &  # FR
            foot_contact[:, 2] &  # RL
            foot_contact[:, 3]    # RR
        ).float()  # [N]

        # only reward when the robot should NOT be moving
        feet_ground_stop = all_feet_grounded * (~should_move).float()      

        # feet dist
        # f_dist_squarred = torch.sum(torch.square(self._robot.data.body_pos_w[:,self._feet_ids[0],:2]-self._robot.data.body_pos_w[:,self._feet_ids[1],:2]), dim = 1)
        # r_dist_squarred = torch.sum(torch.square(self._robot.data.body_pos_w[:,self._feet_ids[2],:2]-self._robot.data.body_pos_w[:,self._feet_ids[3],:2]), dim = 1)
        # feet_dist_error = torch.min((f_dist_squarred - self.cfg.feet_dist_threshold**2) + (r_dist_squarred - self.cfg.feet_dist_threshold**2), torch.zeros(self.num_envs,device=self.device))
        
        # flat 
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1) 
        types  = self._terrain.terrain_types 
        
        stay_flat_mask = ~self.is_on_terrain(["pyramid_stairs", "pyramid_stairs_inv"])
        print(stay_flat_mask)

        # undesired contacts
        # is_contact = (
        #     torch.max(torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        # )
        # contacts = torch.sum(is_contact, dim=1)
        
        # flat orientation
        # flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        
        # stay around default pos:        
        joint_deviation = torch.sum(torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1)
        def_pos = torch.where(should_move, joint_deviation, self.cfg.stand_still_scale * joint_deviation)

        # forces_z = torch.abs(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, 2])
        # forces_xy = torch.linalg.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :2], dim=2)
        # feet_vertical_surface_contacts = torch.any(forces_xy > 4 * forces_z, dim=1).float()
        # feet_vertical_surface_contacts *= torch.clamp(-self._robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        
        # feet to hip distance --------------------------------------------------------------------------------
        
        ROT_W2H = math_utils.matrix_from_quat(math_utils.yaw_quat(self._robot.data.root_quat_w))
        feet_to_base_w = self._robot.data.body_pos_w[:, self._feet_ids, :3] - self._robot.data.root_state_w[:, :3].unsqueeze(1)
        feet_to_base_h = torch.matmul(ROT_W2H.transpose(1,2), feet_to_base_w.transpose(1, 2))
        
        hip_to_base_w = self._robot.data.body_pos_w[:, self._hip_ids, :3] - self._robot.data.root_state_w[:, :3].unsqueeze(1)
        hip_to_base_h = torch.matmul(ROT_W2H.transpose(1,2), hip_to_base_w.transpose(1, 2))
        
        desired_hip_offset = self._desired_hip_offset
        feet_to_hip_distance_x = torch.square(feet_to_base_h[:, 0] - hip_to_base_h[:, 0])
        feet_to_hip_distance_y = torch.square(feet_to_base_h[:, 1] + desired_hip_offset.unsqueeze(0) - hip_to_base_h[:, 1])
        feet_to_hip_distance = -torch.mean(torch.sqrt(feet_to_hip_distance_x + feet_to_hip_distance_y), dim=1)
        # If should_move is False, multiply the distance by 3 (GPU-friendly, vectorized)
        # `should_move` is a boolean tensor defined earlier (shape: [num_envs])
        feet_to_hip_distance = feet_to_hip_distance * torch.where(
            should_move, torch.ones_like(feet_to_hip_distance), torch.full_like(feet_to_hip_distance, 3.0)
        )
        
        
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "track_height": height_error_mapped * self.cfg.height_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "dof_energy_l2": joint_energy * self.cfg.joint_energy_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_rate_2_l2": action_rate_2 * self.cfg.action_rate_2_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "feet_gait": gait * self.cfg.gait_reward_scale * self.step_dt,
            # "feet_dist": feet_dist_error * self.cfg.feet_dist_reward_scale * self.step_dt,
            # "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            # "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "def_pos" : def_pos * self.cfg.def_pos_reward_scale * self.step_dt,
            "feet_to_hip" : feet_to_hip_distance * self.cfg.feet_to_hip_reward_scale * self.step_dt,
            "feet_grounded_stop": feet_ground_stop * self.cfg.feet_grounded_scale * self.step_dt,
            # "feet_vertical_surface" : feet_vertical_surface_contacts * self.cfg.feet_vertical_surface_contacts_reward_scale * self.step_dt
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
        died_base = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id_sensor], dim=-1), dim=1)[0] > 1.0, dim=1)
        died_hips = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._hip_ids_sensor], dim=-1), dim=1)[0] > 1.0, dim=1) 
        died = torch.logical_or(died_base, died_hips)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        reset_env_ids: torch.Tensor = self._robot._ALL_INDICES if env_ids is None else env_ids
        if reset_env_ids.numel() == self.num_envs:
            reset_env_ids = self._robot._ALL_INDICES
        self.curriculum_manager.compute(env_ids=reset_env_ids)
        self._robot.reset(reset_env_ids)
        super()._reset_idx(reset_env_ids)
        if reset_env_ids.numel() == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[reset_env_ids] = 0.0
        self._previous_actions[reset_env_ids] = 0.0
        self._previous_previous_actions[reset_env_ids] = 0.0
        # Sample new commands
        self.command_manager.reset(reset_env_ids)
        if hasattr(self, "_rots"):
            num_resets = reset_env_ids.numel()
            self._rots[reset_env_ids] = torch.empty(num_resets, device=self.device).uniform_(-self.cfg.max_rot, self.cfg.max_rot)
            self._offsets[reset_env_ids] = torch.empty(num_resets, device=self.device).uniform_(
                -self.cfg.max_offset, self.cfg.max_offset
            )
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[reset_env_ids]
        joint_vel = self._robot.data.default_joint_vel[reset_env_ids]
        default_root_state = self._robot.data.default_root_state[reset_env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[reset_env_ids]
        # Add x-axis offset to spawn position
        # default_root_state[:, 0]-= 4.2  # Offset in meters (change this value as needed)
        # default_root_state[:, 1] -= 3.5  # Offset in meters (change this value as needed)
        # # Rotate 45 degrees around z-axis at spawn
        # import math
        # angle = math.pi / 4  # 45 degrees
        # z_rot_quat = torch.tensor(
        #     [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)],
        #     dtype=default_root_state.dtype, device=self.device
        # ).expand(len(env_ids), -1)
        # default_root_state[:, 3:7] = quat_mul(z_rot_quat, default_root_state[:, 3:7])

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], reset_env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], reset_env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, reset_env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][reset_env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][reset_env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[reset_env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[reset_env_ids]).item()
        self.extras["log"].update(extras)
