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
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import CommandManager, CurriculumManager

# Shared with the navigation env and the robot (single implementation of the sensor / observation logic)
from .go2_perception import HeightGrid
from .go2_perception.height_grid import rays_in_lidar_frame
from .go2_perception.locomotion import build_loc_obs, build_proprio
from .go2_perception.sim_noise import (
    CellDropout, LidarBlackout, LidarMisalignment, height_image, sample_lag, set_buffer_lag
)

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
        # self.proprio_buffer_actor = torch.zeros(
        #     self.num_envs, int(self.cfg.proprio_buffer_length), 45, device=self.device
        # )
        # self.proprio_buffer_critic = torch.zeros(
        #     self.num_envs, int(self.cfg.proprio_buffer_length), 48, device=self.device
        # )

        # X/Y linear velocity and yaw angular velocity commands
        self.command_manager = CommandManager(self.cfg.commands, self)
        
        # curriculum manager: keep track of the ability of the robot to follow the command.
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        self._cmd_path = torch.zeros(self.num_envs, device=self.device)
        self._prog_path = torch.zeros(self.num_envs, device=self.device)
        
        self._desired_hip_offset = torch.tensor([-self.cfg.desired_hip_offset, self.cfg.desired_hip_offset, -self.cfg.desired_hip_offset, self.cfg.desired_hip_offset], device=self.device)
        if self.cfg.delay:
            # Same convention as the navigation env: cfg.delay_length, lags uniform in [0, delay_length] control steps,
            # independent lag for proprio and for the height grid (separate sensors).
            L = int(self.cfg.delay_length)
            self._buffer = DelayBuffer(history_length=L, batch_size=self.num_envs, device=self.device)
            self._grid_buffer = DelayBuffer(history_length=L, batch_size=self.num_envs, device=self.device)
            set_buffer_lag(self._buffer, sample_lag(self.num_envs, L, self.device))
            set_buffer_lag(self._grid_buffer, sample_lag(self.num_envs, L, self.device))
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
                "feet_long_air",
                "feet_air_stand",
                "feet_gait",
                # "feet_dist",
                "undesired_contacts",
                "flat_orientation_l2",
                "pitch_orientation_l1",
                "def_pos",
                "feet_to_hip",
                "feet_grounded_stop",
                "feet_vertical_surface",
                "heading_error",
                "progress",
                # "termination",
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
        
        self._undesired_contact_body_ids_sensor = self._thigh_ids_sensor + self._hip_ids_sensor + self._base_id_sensor
        self._body_contact_info_teacher_sensor = self._base_id_sensor + self._thigh_ids_sensor + self._calf_ids_sensor
        self._finite_warn_counter = 0

        # Total lidar outages at random moments of the episode (sim only). Optional cfg fields:
        #   lidar_blackout (bool, default True), blackout_fill_value (float, default 0.0), cell_dropout (bool, default True)
        self.blackout = None
        if isinstance(self.cfg, Go2LidarRoughEnvCfg) and getattr(self.cfg, "lidar_blackout", True):
            self.blackout = LidarBlackout(
                self.num_envs, self.max_episode_length, getattr(self.cfg, "blackout_fill_value", 0.0), self.device
            )
        
        
        # self._step_freq = torch.tensor(self.cfg.desired_step_freq, device=self.device)
        # self._duty_factor = torch.tensor(self.cfg.desired_duty_factor, device=self.device)
        # self._phase_offset = torch.tensor(self.cfg.desired_phase_offset, device=self.device).repeat(self.num_envs,1)
        # self._phase_signal = self._phase_offset.clone()
        # self._phase_signal = self._phase_signal % 1.0
        
    
    def build_col_to_subterrain(self):
        gen = self.scene.terrain.cfg.terrain_generator
        names = list(gen.sub_terrains.keys())
        props = torch.tensor([gen.sub_terrains[n].proportion for n in names])
        cum = torch.cumsum(props / props.sum(), 0)
        col_to_name = {}
        for col in range(gen.num_cols):
            idx = int((col / gen.num_cols + 0.001 < cum).nonzero()[0])
            col_to_name[col] = names[idx]
        return col_to_name
    
    def build_terrain_mask(self):
        for terrain_name in self.scene.terrain.cfg.terrain_generator.sub_terrains.keys():
            col_to_bool = torch.zeros(self.scene.terrain.cfg.terrain_generator.num_cols, dtype=torch.bool, device=self.scene.terrain.device)
            for col, name in self.build_col_to_subterrain().items():
                col_to_bool[col] = (name == terrain_name)
            # Index by terrain_types once — this never changes
            self._terrain_masks[terrain_name] = col_to_bool[self.scene.terrain.terrain_types]

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

    def _setup_scene(self):
        if self.cfg.vis:
            self.env_markers = VisualizationMarkers(self.cfg.env_marker_cfg)
            self.vis_envs: torch.Tensor = torch.tensor([13])
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

            # height grid (same class as the navigation env / robot) and sim-only cell dropout
            self.height_grid = HeightGrid(
                self.cfg.x_range, self.cfg.y_range, self.cfg.res, self.cfg.desired_base_height, self.device
            )
            self.dropout = CellDropout(
                self.height_grid, None, self.cfg.sigma, self.cfg.n_zeros, self.cfg.max_reset_zeros_freq, self.device
            )
            # lidar mounting error (rotations + 3D ray offset), identical to the navigation env
            self.lidar_noise = LidarMisalignment(self.num_envs, self.cfg.max_rot, self.cfg.max_offset, self.device)
            # self._rots = torch.tensor()
            

 
        self._subterrain_names = list(self.scene.terrain.cfg.terrain_generator.sub_terrains.keys())
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

    # ------------------------------------------------------------------ height sensor
    def _lidar_rays(self) -> torch.Tensor:
        """Clean ray hits in the lidar frame (policy convention). Computed once per step."""
        data = self._height_scanner.data
        return rays_in_lidar_frame(data.ray_hits_w, data.pos_w, data.quat_w)

    def is_on_terrain(self, terrain_names: list[str]) -> torch.Tensor:
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name in terrain_names:
            mask |= self._terrain_masks[name]
        return mask
    
    def _get_observations(self) -> dict:
        
        # Unified observation construction for all env variants.
        # Sensor model identical to the navigation env: misaligned rays -> grid (+z noise) -> sanitize -> delay.
        randomize = bool(self.cfg.randomize)
        rays_clean = self._lidar_rays()
        rays_actor = self.lidar_noise.apply(rays_clean) if (randomize and hasattr(self, "lidar_noise")) else rays_clean

        # critic: clean, privileged
        height_data, valid_critic = height_image(
            self.height_grid, rays_clean, self.dropout, noise=0.0, apply_dropout=False
        )
        # actor: noisy; partial cell dropout only when randomizing (as in the navigation env)
        height_data_actor, valid_actor = height_image(
            self.height_grid, rays_actor, self.dropout, noise=0.01 if randomize else 0.0,
            apply_dropout=randomize and getattr(self.cfg, "cell_dropout", True),
        )
        # total lidar outage: heights = preset value, valid = False, for the envs inside their blackout window
        if randomize and self.blackout is not None:
            height_data_actor, valid_actor, blackout_active = self.blackout.apply(
                height_data_actor, valid_actor, self.episode_length_buf
            )
            self.extras.setdefault("log", {})["Debug/lidar_blackout_frac"] = blackout_active.float().mean()

        height_data = self._sanitize_tensor(height_data, "height_data", clamp_abs=10.0)
        height_data_actor = self._sanitize_tensor(height_data_actor, "height_data_actor", clamp_abs=10.0)
        valid_critic = valid_critic.float()
        valid_actor = valid_actor.float()
        if self.cfg.vis:
            torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
            print(height_data[self.vis_envs] + self.cfg.desired_base_height)
            print(height_data[self.vis_envs].mean())
        actor_grid = torch.stack([height_data_actor, valid_actor], dim=1)     # (N, 2, x_cells, y_cells)
        critic_grid = torch.stack([height_data, valid_critic], dim=1)

        actor_proprio_noisy = build_proprio(
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            noise_scale=float(randomize),
        )
        actor_proprio_noisy = self._sanitize_tensor(actor_proprio_noisy, "actor_proprio", clamp_abs=50.0)

        # sensor latency: proprio and height grid each go through their own delay buffer
        actor_proprio_delayed = actor_proprio_noisy
        if self.cfg.delay:
            actor_proprio_delayed = self._buffer.compute(actor_proprio_noisy)
            actor_grid = self._grid_buffer.compute(actor_grid)

        # command and last action are not delayed (same as the navigation env)
        actor_proprio = build_loc_obs(
            actor_proprio_delayed,
            self.command_manager.get_command("base_velocity"),
            self._actions,
        )
        # Critic (privileged) proprio observations — include privileged sensors like base linear velocity,
        # contact flags and other privileged terms useful for value estimation.
        
        critic_proprio = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self.command_manager.get_command("base_velocity"),
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
                # clock_data
            ],
            dim=-1,
        )
        critic_proprio = self._sanitize_tensor(critic_proprio, "critic_proprio", clamp_abs=100.0)
        # self._update_proprio_buffers(actor_proprio, critic_proprio)
        
        # Update previous actions and return unified observation dict (flat keys for runner grouping).
        self._previous_actions = self._actions.clone()
        
        return {
            "actor_proprio": actor_proprio,
            "actor_grid": actor_grid,
            "critic_proprio": critic_proprio,
            "critic_grid": critic_grid,
        }

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        if self.cfg.vis:
            self.env_markers.visualize(
                translations=self._robot.data.root_pos_w[self.vis_envs],
                orientations=self._robot.data.root_quat_w[self.vis_envs],
            )
        if not self.cfg.test:
            terrain_mask = (~self.is_on_terrain(["pyramid_stairs_25", "pyramid_stairs_inv_25", "pyramid_stairs_30", "pyramid_stairs_inv_30", "pyramid_stairs_35", "pyramid_stairs_inv_35", "pyramid_stairs_40", "pyramid_stairs_inv_40", ])).float()
            # terrain_mask = (~self.is_on_terrain(["pyramid_stairs", "pyramid_stairs_inv"])).float()
        else: 
            terrain_mask = True
        # terrain_mask = (~self.is_on_terrain(["pyramid_stairs", "pyramid_stairs_inv"])).float()
        term = self.command_manager.get_term("base_velocity")
        cmd = self.command_manager.get_command("base_velocity")
        cmd_xy = cmd[:, :2]
        
        #heading error (prevent robot from turning around in stairs climbing)
        heading_err = torch.abs(math_utils.wrap_to_pi(term.heading_target - self._robot.data.heading_w))
        cmd_full = torch.linalg.norm(cmd, dim=1) 
        heading_rew = torch.exp(-torch.square(heading_err) / 0.25) * term.is_heading_env.float() * (cmd_full > 0.01).float()
        # utils for the curriculum
        ref_yaw = torch.where(term.is_heading_env, term.heading_target, self._robot.data.heading_w)
        c, s = torch.cos(ref_yaw), torch.sin(ref_yaw)
        cmd_w = torch.stack(
            (c * cmd_xy[:, 0] - s * cmd_xy[:, 1],
            s * cmd_xy[:, 0] + c * cmd_xy[:, 1]),
            dim=1,
        )
        v_w = self._robot.data.root_lin_vel_w[:, :2]
        cmd_w_norm = torch.norm(cmd_w, dim=1)
        dir_w = cmd_w / cmd_w_norm.unsqueeze(1).clamp(min=1e-6)
        proj = (v_w * dir_w).sum(dim=1)
        # curriculum: forward progress only, no credit for overshooting
        proj_pos = torch.minimum(proj.clamp(min=0.0), cmd_w_norm)
        self._cmd_path += cmd_w_norm * self.step_dt
        self._prog_path += proj_pos * self.step_dt

        # reward in [-1, 1]; moving away from the target direction is penalized
        proj_clipped = torch.maximum(torch.minimum(proj, cmd_w_norm), -cmd_w_norm)
        progress = proj_clipped / cmd_w_norm.clamp(min=1e-6)
        progress = progress * (cmd_w_norm > 0.1).float()
        
        # lin vel error
        lin_vel_error = torch.sum((cmd_w - v_w)**2, dim=1)
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
        height_error_mapped = torch.exp(-height_error / 0.01) * terrain_mask
        
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
        air_time = torch.sum(
            (last_air_time.clamp(max=0.4) - 0.2).clamp(min=0.0) * first_contact, dim=1
        ) * (torch.norm(cmd_xy, dim=1) > 0.1)

        # penalty for a foot that stays airborne too long
        current_air = self._contact_sensor.data.current_air_time[:, self._feet_ids_sensor]
        long_air = (current_air - 0.5).clamp(min=0.0).sum(dim=1)
                
        # useful for next two rewards:
        cmd = torch.linalg.norm(self.command_manager.get_command("base_velocity"), dim=1)
        should_move = cmd > 0.01

        foot_contact = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor], dim=-1) > 1.0
        # foot_contact_hist = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids_sensor, :].norm(dim=-1).max(dim=1)[0] > 1.0

        # gait trot
        # self._phase_signal += self.step_dt * self._step_freq
        # self._phase_signal = self._phase_signal % 1.0
        # contact_periodic_on = self._phase_signal < self._duty_factor
        # gait = (torch.sum(contact_periodic_on*foot_contact_hist, dim=1) + \
                                #    torch.sum(~contact_periodic_on*~foot_contact_hist, dim=1))*should_move/4.0


        # gait trot
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
        moving_xy = torch.norm(cmd_xy, dim=1) > 0.1
        gait = (trot_pattern - 0.5 * wrong_pair).clamp(min=0.0) * moving_xy
        
        # all feet grounded when stopped
        all_feet_grounded = (
            foot_contact[:, 0] &  # FL
            foot_contact[:, 1] &  # FR
            foot_contact[:, 2] &  # RL
            foot_contact[:, 3]    # RR
        ).float()  # [N]

        # only reward when the robot should NOT be moving
        feet_ground_stop = all_feet_grounded * (~should_move).float() * terrain_mask  
        stand = ~(cmd > 0.01)
        feet_air_stand = (~foot_contact).float().sum(dim=1) * stand.float()    

        # feet dist
        # f_dist_squarred = torch.sum(torch.square(self._robot.data.body_pos_w[:,self._feet_ids[0],:2]-self._robot.data.body_pos_w[:,self._feet_ids[1],:2]), dim = 1)
        # r_dist_squarred = torch.sum(torch.square(self._robot.data.body_pos_w[:,self._feet_ids[2],:2]-self._robot.data.body_pos_w[:,self._feet_ids[3],:2]), dim = 1)
        # feet_dist_error = torch.min((f_dist_squarred - self.cfg.feet_dist_threshold**2) + (r_dist_squarred - self.cfg.feet_dist_threshold**2), torch.zeros(self.num_envs,device=self.device))
        
        # flat 
        flat_l2 = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1) 
        flat_terrain_l2 = terrain_mask * flat_l2
        
        # pitch l1
        pitch_l1 = self._robot.data.projected_gravity_b[:, 0]
        dead_zone = 0.05
        pitch_terrain_l1 = (pitch_l1.abs() - dead_zone).clamp(min=0.0) * terrain_mask
        
        # undesired contacts
        is_contact = (
            torch.max(torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids_sensor], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        
        # flat orientation
        # flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        
        # stay around default pos:        
        joint_deviation = torch.sum(torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1)
        def_pos = torch.where(should_move, joint_deviation, self.cfg.stand_still_scale * joint_deviation) * terrain_mask
        
        forces_z = torch.abs(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, 2])
        forces_xy = torch.linalg.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :2], dim=2)
        feet_vertical_surface_contacts = torch.any(forces_xy > 4 * forces_z, dim=1).float()
        feet_vertical_surface_contacts *= torch.clamp(-self._robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        
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
        
        # termination penalty
        # termination = self.reset_terminated.float()
        
        
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
            "feet_long_air": long_air * self.cfg.feet_long_air_reward_scale * self.step_dt,
            "feet_gait": gait * self.cfg.gait_reward_scale * self.step_dt,
            "feet_air_stand": feet_air_stand * self.cfg.feet_air_stand_reward_scale * self.step_dt,
            # "feet_dist": feet_dist_error * self.cfg.feet_dist_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_terrain_l2 * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "pitch_orientation_l1": pitch_terrain_l1 * self.cfg.pitch_orientation_reward_scale * self.step_dt,
            "def_pos" : def_pos * self.cfg.def_pos_reward_scale * self.step_dt,
            "feet_to_hip" : feet_to_hip_distance * self.cfg.feet_to_hip_reward_scale * self.step_dt,
            "feet_grounded_stop": feet_ground_stop * self.cfg.feet_grounded_scale * self.step_dt,
            "feet_vertical_surface" : feet_vertical_surface_contacts * self.cfg.feet_vertical_surface_contacts_reward_scale * self.step_dt,
            "heading_error": heading_rew * self.cfg.heading_rew_reward_scale * self.step_dt,
            "progress": progress * self.cfg.progress_reward_scale * self.step_dt,
            # "termination": termination * self.cfg.termination_reward_scale,
        }
        standing = cmd < 0.01
        self.extras["log"]["Debug/heading_err_standing"] = heading_err[standing].mean()
        self.extras["log"]["Debug/gait_rew_standing"] = gait[standing].mean()
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
        
        # curriculum
        self.curriculum_manager.compute(env_ids=reset_env_ids)
        self._cmd_path[reset_env_ids] = 0.0
        self._prog_path[reset_env_ids] = 0.0
        
        super()._reset_idx(reset_env_ids)
        if reset_env_ids.numel() == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            
        self._actions[reset_env_ids] = 0.0
        self._previous_actions[reset_env_ids] = 0.0
        self._previous_previous_actions[reset_env_ids] = 0.0
        # Sample new commands
        self.command_manager.reset(reset_env_ids)
        term = self.command_manager.get_term("base_velocity")
        upper = 10.0
        min_sample_range = 2.0
        lower = torch.empty(len(reset_env_ids), device=self.device).uniform_(min_sample_range, upper)  # one lower bound per env
        # time_left_i ~ U(lower_i, upper)
        term.time_left[env_ids] = lower + (upper - lower) * torch.rand(len(reset_env_ids), device=self.device)
        if hasattr(self, "lidar_noise"):
            self.lidar_noise.resample(reset_env_ids)
        if self.blackout is not None:
            self.blackout.resample(reset_env_ids)

        if self.cfg.delay:
            L = int(self.cfg.delay_length)
            n = reset_env_ids.numel()
            self._buffer.reset(reset_env_ids)
            set_buffer_lag(self._buffer, sample_lag(n, L, self.device), batch_ids=reset_env_ids)
            self._grid_buffer.reset(reset_env_ids)
            set_buffer_lag(self._grid_buffer, sample_lag(n, L, self.device), batch_ids=reset_env_ids)

        # reset phase
        # self._phase_signal[env_ids] = self._phase_offset[env_ids].clone()
        # self._phase_signal[env_ids] = self._phase_signal[env_ids]  % 1.0
        
        # Reset buffers
        # self.proprio_buffer_actor[reset_env_ids] = 0.0
        # self.proprio_buffer_critic[reset_env_ids] = 0.0
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][reset_env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][reset_env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Curriculum/terrain_level"] = self.scene.terrain.terrain_levels.float().mean()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[reset_env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[reset_env_ids]).item()
        self.extras["log"].update(extras)



    def _update_proprio_buffers(self, proprio_obs_actor: torch.Tensor, proprio_obs_critic: torch.Tensor) -> None:
        """
        updating odom_obs_hist, filled by the back: the most recent are at the end of the buffer.
        """
        self.proprio_buffer_actor = torch.roll(self.proprio_buffer_actor, -1, 1)
        self.proprio_buffer_actor[:, -1, :] = proprio_obs_actor
        self.proprio_buffer_critic = torch.roll(self.proprio_buffer_critic, -1, 1)
        self.proprio_buffer_critic[:, -1, :] = proprio_obs_critic