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
from .go2_perception.sim_noise import height_image
from .go2_perception.locomotion import build_loc_obs, build_proprio
from isaaclab.utils.math import quat_apply_inverse   # older Isaac Lab versions: quat_rotate_inverse

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
        self.build_terrain_mask()
        N, J = self.num_envs, self._robot.num_joints
        dev = self.device
        self._all_joint_ids = torch.arange(J, device=dev)
        self._rand_mass = torch.zeros(N, 1, device=dev)
        self._rand_com = torch.zeros(N, 3, device=dev)
        self._rand_friction = torch.ones(N, 1, device=dev)
        self._rand_restitution = torch.zeros(N, 1, device=dev)
        self._rand_kp = torch.zeros(N, J, device=dev)
        self._rand_kd = torch.zeros(N, J, device=dev)
        self._rand_push = torch.zeros(N, 6, device=dev)

        # nominal COM = median over envs (the randomization is symmetric around it)
        coms = self._robot.root_physx_view.get_coms()
        self._nominal_com = coms[:, self._base_id, :3].median(dim=0).values.to(dev)

        # shape indices that belong to the feet (same logic Isaac Lab uses in randomize_rigid_body_material)
        foot_body_ids = self._robot.find_bodies(".*_foot")[0]
        view = self._robot.root_physx_view
        n_shapes = []
        for link_path in view.link_paths[0]:
            n_shapes.append(self._robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes)
        starts = torch.cumsum(torch.tensor([0] + n_shapes[:-1]), 0)
        self._foot_shape_ids = torch.cat(
            [torch.arange(starts[b], starts[b] + n_shapes[b]) for b in foot_body_ids]
        )

        self._read_randomization()   # startup events already ran
        
    def _read_randomization(self, env_ids=None):
        """Pull the randomized physics params back from the sim into teacher buffers."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        robot = self._robot
        view = robot.root_physx_view
        cpu_ids = env_ids.cpu()

        # --- base mass offset (startup, "add") ---
        masses = view.get_masses()                                   # (N, num_bodies) CPU
        base_id = self._base_id
        self._rand_mass[env_ids, 0] = (
            masses[cpu_ids, base_id] - robot.data.default_mass[cpu_ids, base_id]
        ).to(self.device)

        # --- base COM (startup) ---
        coms = view.get_coms()                                       # (N, num_bodies, 7): pos + quat
        com_pos = coms[cpu_ids, base_id, :3].to(self.device)
        self._rand_com[env_ids] = com_pos - self._nominal_com        # centered on the nominal COM

        # --- feet friction / restitution (startup) ---
        mat = view.get_material_properties()                         # (N, num_shapes, 3) CPU: static, dynamic, restitution
        foot_mat = mat[cpu_ids][:, self._foot_shape_ids]             # (n, n_foot_shapes, 3)
        self._rand_friction[env_ids, 0] = foot_mat[..., 0].mean(-1).to(self.device)
        self._rand_restitution[env_ids, 0] = foot_mat[..., 2].mean(-1).to(self.device)

        # --- actuator gains (reset, "scale") ---
        for act in robot.actuators.values():
            idx = act.joint_indices
            idx = slice(None) if isinstance(idx, slice) and idx == slice(None) else idx
            kp_nom = robot.data.default_joint_stiffness[env_ids][:, idx]
            kd_nom = robot.data.default_joint_damping[env_ids][:, idx]
            jid = self._all_joint_ids[idx]                           # joint ids covered by this actuator group
            self._rand_kp[env_ids[:, None], jid] = act.stiffness[env_ids] / kp_nom - 1.0
            self._rand_kd[env_ids[:, None], jid] = act.damping[env_ids] / kd_nom - 1.0
        
    def get_terrain_one_hot(self) -> torch.Tensor:
        return self._terrain_one_hot  # (num_envs, num_terrain_types)
        
    def build_terrain_mask(self):
        self._terrain_masks = {}
        names = list(self.scene.terrain.cfg.terrain_generator.sub_terrains.keys())
        col_to_name = self.build_col_to_subterrain()

        for terrain_name in names:
            col_to_bool = torch.zeros(
                self.scene.terrain.cfg.terrain_generator.num_cols,
                dtype=torch.bool, device=self.scene.terrain.device,
            )
            for col, name in col_to_name.items():
                col_to_bool[col] = (name == terrain_name)
            self._terrain_masks[terrain_name] = col_to_bool[self.scene.terrain.terrain_types]

        # Fixed ordering -> consistent one-hot indices
        self._terrain_names = names
        # (num_envs, num_terrain_types), float so it can be cat'd with obs
        self._terrain_one_hot = torch.stack(
            [self._terrain_masks[n] for n in names], dim=-1
        ).float()
        assert torch.all(self._terrain_one_hot.sum(-1) == 1)

    def _get_observations(self) -> dict:
        # Unified observation construction for all env variants.
        # Sensor model identical to the navigation env: misaligned rays -> grid (+z noise) -> sanitize -> delay.
        randomize = bool(self.cfg.randomize)
        rays_clean = self._lidar_rays()
        rays_student = self.lidar_noise.apply(rays_clean) if (randomize and hasattr(self, "lidar_noise")) else rays_clean
        
        # teacher: clean, privileged
        height_data_teacher, valid_teacher = height_image(
            self.height_grid, rays_clean, self.dropout, noise=0.0, apply_dropout=False
        )
        # actor: noisy; partial cell dropout only when randomizing (as in the navigation env)
        height_data_student, valid_student = height_image(
            self.height_grid, rays_student, self.dropout, noise=0.01 if randomize else 0.0,
            apply_dropout=randomize and getattr(self.cfg, "cell_dropout", True),
        )
        # total lidar outage: heights = preset value, valid = False, for the envs inside their blackout window
        if randomize and self.blackout is not None:
            height_data_student, valid_student, blackout_active = self.blackout.apply(
                height_data_student, valid_student, self.episode_length_buf
            )
            self.extras.setdefault("log", {})["Debug/lidar_blackout_frac"] = blackout_active.float().mean()
        
        height_data_teacher = self._sanitize_tensor(height_data_teacher, "height_data", clamp_abs=10.0)
        height_data_student = self._sanitize_tensor(height_data_student, "height_data_actor", clamp_abs=10.0)
        valid_teacher = valid_teacher.float()
        valid_student = valid_student.float()
        if self.cfg.vis:
            torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
            print(height_data_teacher[self.vis_envs] + self.cfg.desired_base_height)
            print(height_data_teacher[self.vis_envs].mean())
        grid_student = torch.stack([height_data_student, valid_student], dim=1)     # (N, 2, x_cells, y_cells)
        grid_teacher = torch.stack([height_data_teacher, valid_teacher], dim=1)
        
        proprio_student = build_proprio(
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            noise_scale=float(randomize),
        )
        proprio_student = self._sanitize_tensor(proprio_student, "actor_proprio", clamp_abs=50.0)
        
        # sensor latency: proprio and height grid each go through their own delay buffer
        proprio_student_delayed = proprio_student
        if self.cfg.delay:
            proprio_student_delayed = self._buffer.compute(proprio_student)
            grid_student = self._grid_buffer.compute(grid_student)
        
        # command and last action are not delayed (same as the navigation env)
        student_frame = build_loc_obs(
            proprio_student_delayed,
            self.command_manager.get_command("base_velocity"),
            self._actions,
        )
        proprio_student = self.proprio_hist.push(student_frame)  # [N, (H+1) * 45]
        # ---- Teacher (privileged) proprio -------------------------------------------------
        s = self.cfg.obs_scales
        d = self._robot.data
        cs = self.scene.sensors["contact_forces"]
        fid = self._feet_ids                                   # (4,) from cs.find_bodies(".*_foot")

        # base height above terrain, from the clean grid (heights are relative to desired_base_height)
        base_height = (height_data_teacher + self.cfg.desired_base_height).mean(dim=(1, 2), keepdim=True).squeeze(-1)  # (N,1)

        # contacts
        feet_forces = cs.data.net_forces_w[:, fid, :]                              # (N,4,3)
        feet_force_mag = feet_forces.norm(dim=-1)                                  # (N,4)
        feet_contact = (feet_force_mag > 1.0).float()                              # (N,4)
        air_time = cs.data.current_air_time[:, fid].clamp(max=1.0)                 # (N,4)  needs track_air_time=True

        # feet kinematics in the base frame
        feet_pos_b = quat_apply_inverse(
            d.root_quat_w.unsqueeze(1).expand(-1, len(fid), -1),
            d.body_pos_w[:, fid] - d.root_pos_w.unsqueeze(1),
        )                                                                          # (N,4,3)
        feet_vel_b = quat_apply_inverse(
            d.root_quat_w.unsqueeze(1).expand(-1, len(fid), -1),
            d.body_lin_vel_w[:, fid] - d.root_lin_vel_w.unsqueeze(1),
        )                                                                          # (N,4,3)
        # foot clearance above the terrain below the base
        terrain_z = d.root_pos_w[:, 2:3] - base_height                             # (N,1)
        feet_height = d.body_pos_w[:, fid, 2] - terrain_z                          # (N,4)

        # terrain difficulty
        terr = self.scene.terrain
        terrain_level = (terr.terrain_levels.float() / max(terr.max_terrain_level - 1, 1)).unsqueeze(-1)  # (N,1)

        proprio_teacher = torch.cat(
            [
                d.root_lin_vel_b * s.lin_vel,
                d.root_ang_vel_b * s.ang_vel,
                d.projected_gravity_b * s.gravity,
                self.command_manager.get_command("base_velocity") * self._cmd_scale,   # tensor [2,2,0.25]
                (d.joint_pos - d.default_joint_pos) * s.joint_pos,
                d.joint_vel * s.joint_vel,
                d.joint_acc * s.joint_acc,
                d.applied_torque * s.torque,
                self._actions * s.action,
                self._previous_actions * s.action,
                # terrain
                self.get_terrain_one_hot(),
                terrain_level,
                base_height * s.height,
                # contacts / feet
                feet_contact,
                feet_force_mag * s.contact_force,
                air_time * s.air_time,
                feet_pos_b.flatten(1) * s.feet_pos,
                feet_vel_b.flatten(1) * s.feet_vel,
                feet_height * s.height,
                # randomization params (see section 2)
                self._rand_friction * s.friction,          # (N,1)
                self._rand_restitution * s.restitution,    # (N,1)
                self._rand_mass * s.mass,                  # (N,1)
                self._rand_com * s.com,                    # (N,3)
                self._rand_kp * s.gain,                    # (N,J)  already centered at 0
                self._rand_kd * s.gain, 
                # clock_data
            ],
            dim=-1,
        )
        proprio_teacher = self._sanitize_tensor(proprio_teacher, "teacher_proprio", clamp_abs=100.0)
        
        return {
            "student_proprio": proprio_student,
            "student_grid": grid_student,
            "teacher_proprio": proprio_teacher,
            "teacher_grid": grid_teacher,
        }

class Go2TeacherStudentCNNEnv(Go2LidarEnv):
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
        """
        Return both student and teacher observations for distillation.
        """
        # height data (laser)
        C, H, W = 1, 15, 10
        # Previous height logic kept for reference:
        # height_data = self._compute_height_data("normal")
        height_data = self._compute_height_data_from_cloud() - 0.5
        height_data_student = height_data + (2.0 * torch.rand_like(height_data) - 1.0) * float(0.01) * self.cfg.randomize
        height_data_student = self._randomize_heightmap(height_data_student) 
        height_data = height_data.view(self.num_envs, C, H, W)
        height_data_student = height_data_student.view(self.num_envs, C, H, W) 
        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        # print(height_data+0.5)   
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        velocity_commands = self.command_manager.get_command("base_velocity")
        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        # cell_size_m = float(self.cfg.res)
        # inv_cell_size = 1.0 / cell_size_m
        # x_min, x_max = float(self.cfg.x_range[0]), float(self.cfg.x_range[1])
        # y_min, y_max = float(self.cfg.y_range[0]), float(self.cfg.y_range[1])
        # print(height_data_student.reshape(int((x_max - x_min)*inv_cell_size),int((y_max - y_min)*inv_cell_size)))
        
        # print(height_data.reshape(self.num_envs, 15, 10).flip(1,2))            
            
        
        proprio_student = torch.cat(
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
        
        base_lin_vel = self._robot.data.root_lin_vel_b 
        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor], dim=-1) > 1.0).float()
        is_contact = (
            torch.max(torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._body_contact_info_teacher_sensor], dim=-1), dim=1)[0] > 1.0
        )

        proprio_teacher = torch.cat(
            [
                base_ang_vel,      
                projected_gravity, 
                velocity_commands,       
                joint_pos_rel,          
                joint_vel,          
                self._actions, 
                base_lin_vel,                           
                foot_contacts,     
                is_contact,
            ],
            dim=-1,
        )
        teacher_height_scan = self._sanitize_tensor(height_data, "height_data", clamp_abs=10.0)
        student_height_scan = self._sanitize_tensor(height_data_student, "height_data_student", clamp_abs=10.0)
        teacher_proprio = self._sanitize_tensor(proprio_teacher, "teacher_obs", clamp_abs=100.0)
        student_proprio = self._sanitize_tensor(proprio_student, "proprio_student", clamp_abs=100.0)
        
        self._previous_actions = self._actions.clone()
        
        return {
            "student_proprio": student_proprio,
            "student_height_scan": student_height_scan,
            "teacher_proprio": teacher_proprio,
            "teacher_height_scan": teacher_height_scan,
        }

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)          # this applies the "reset" events (joint gains, etc.)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self._read_randomization(env_ids)
        self._rand_push[env_ids] = 0.0

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
        # Previous height logic kept for reference:
        # height_data = self._compute_height_data("normal")
        height_data = self._compute_height_data_from_cloud()
        height_data_student = height_data + (2.0 * torch.rand_like(height_data) - 1.0) * float(0.01) * self.cfg.randomize
        height_data_student = self._randomize_heightmap(height_data_student) 
        height_data = self._sanitize_tensor(height_data, "height_data", clamp_abs=10.0)
        height_data_student = self._sanitize_tensor(height_data_student, "height_data_actor", clamp_abs=10.0)
        
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b 
        joint_pos_rel = (self._robot.data.joint_pos - self._robot.data.default_joint_pos) 
        joint_vel = self._robot.data.joint_vel    
        velocity_commands = self.command_manager.get_command("base_velocity")
        self._previous_actions = self._actions.clone()
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
        student_obs = self._sanitize_tensor(student_obs, "student_obs", clamp_abs=100.0)

        return {"policy": student_obs}
