# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# JOINT ORDER:
#   - "FL_hip_joint"
#   - "FR_hip_joint"
#   - "RL_hip_joint"
#   - "RR_hip_joint"
#   - "FL_thigh_joint"
#   - "FR_thigh_joint"
#   - "RL_thigh_joint"
#   - "RR_thigh_joint"
#   - "FL_calf_joint"
#   - "FR_calf_joint"
#   - "RL_calf_joint"
#   - "RR_calf_joint"


import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the raycaster sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

# Set torch print options
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_conjugate

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip
import isaaclab.terrains as terrain_gen
from isaaclab.utils.math import subtract_frame_transforms, quat_inv, quat_apply
from isaaclab.terrains.config import TerrainGeneratorCfg


##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG 

MAX_RAY_DIST = 1.0 # the rays make a circle of radius = sqrt(2) * MAX_RAY_DIST, so that the height map is fully 
OFFSET = (0.28945, 0.0, -0.04682) # offset for the lidar from the base
ROOM_CENTER_X = 0.0
ROOM_CENTER_Y = 4.0
ROOM_INNER_SIZE = 10.0
ROOM_WALL_THICKNESS = 0.1
ROOM_WALL_HEIGHT = 5.0
W, X, Y, Z = quat_from_euler_xyz(torch.tensor(-torch.pi), torch.tensor(torch.pi - 2.8782), torch.tensor(-torch.pi))
@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # stairs terrain for lidar debugging
    ROUGH_TERRAINS_CFG.num_cols = 1
    ROUGH_TERRAINS_CFG.num_rows = 2
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=1,
            num_cols=2,
            curriculum=True,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.5,
                    step_height_range=(0.05, 0.15),
                    step_width=0.3,
                    platform_width=2.0,
                    border_width=1.0,
                    holes=False,
                ),
                "room_floor": terrain_gen.MeshPlaneTerrainCfg(
                    proportion=0.5,
                ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # four static walls for a simple room next to the pyramid tile
    room_wall_north = AssetBaseCfg(
        prim_path="/World/room/wall_north",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                ROOM_CENTER_X,
                ROOM_CENTER_Y + ROOM_INNER_SIZE / 2.0 + ROOM_WALL_THICKNESS / 2.0,
                ROOM_WALL_HEIGHT / 2.0,
            )
        ),
        spawn=sim_utils.CuboidCfg(
            size=(ROOM_INNER_SIZE + 2.0 * ROOM_WALL_THICKNESS, ROOM_WALL_THICKNESS, ROOM_WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    room_wall_south = AssetBaseCfg(
        prim_path="/World/room/wall_south",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                ROOM_CENTER_X,
                ROOM_CENTER_Y - ROOM_INNER_SIZE / 2.0 - ROOM_WALL_THICKNESS / 2.0,
                ROOM_WALL_HEIGHT / 2.0,
            )
        ),
        spawn=sim_utils.CuboidCfg(
            size=(ROOM_INNER_SIZE + 2.0 * ROOM_WALL_THICKNESS, ROOM_WALL_THICKNESS, ROOM_WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    room_wall_east = AssetBaseCfg(
        prim_path="/World/room/wall_east",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                ROOM_CENTER_X + ROOM_INNER_SIZE / 2.0 + ROOM_WALL_THICKNESS / 2.0,
                ROOM_CENTER_Y,
                ROOM_WALL_HEIGHT / 2.0,
            )
        ),
        spawn=sim_utils.CuboidCfg(
            size=(ROOM_WALL_THICKNESS, ROOM_INNER_SIZE, ROOM_WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    room_wall_west = AssetBaseCfg(
        prim_path="/World/room/wall_west",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                ROOM_CENTER_X - ROOM_INNER_SIZE / 2.0 - ROOM_WALL_THICKNESS / 2.0,
                ROOM_CENTER_Y,
                ROOM_WALL_HEIGHT / 2.0,
            )
        ),
        spawn=sim_utils.CuboidCfg(
            size=(ROOM_WALL_THICKNESS, ROOM_INNER_SIZE, ROOM_WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # robot
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # specs of the real lidar:
    '''
    360 * 90 degrees FOV
    translated by approx 0.3 m on x axis and -0.047 along z axis
    rotated by approx. (rpy) [-180, 15.1, -180]. (15.1 degrees along yaw !!!) 
    The values have been chosen debugging the vis 
    '''
    # ray_caster = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base",
    #     # offset=RayCasterCfg.OffsetCfg(pos=(0.28945 + 0.25, 0.0, -0.04682)),
    #         offset=RayCasterCfg.OffsetCfg(pos=(0.28945 + 0.25, 0.0, 0.5)),
    #     update_period=1/20,
    #     ray_alignment="base",
    #     attach_yaw_only=False,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.4, 0.9], ordering = "yx"),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    ray_caster = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=OFFSET,
            rot=(W, X, Y, Z),  
        ),
        # Keep a single target rooted at /World; MultiMeshRayCaster will include all
        # supported meshes/shapes below it (including the room wall cubes).
        mesh_prim_paths=["/World"],
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=64, vertical_fov_range=[-0.0, 90.0], horizontal_fov_range=[-180, 180], horizontal_res=2.0
        ),
        max_distance= 4.0  ,
        debug_vis=not args_cli.headless,
    )


# Precompute Gaussian probability grid (since h=15, w=10 are fixed)
_gaussian_prob_cache = {}

def _compute_gaussian_prob(h: int, w: int, sigma: float, device) -> torch.Tensor:
    """Compute and cache Gaussian probability distribution."""
    cache_key = (h, w, sigma, device)
    if cache_key not in _gaussian_prob_cache:
        # Create coordinate grids (grid indices)
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Center of the grid
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        
        # Compute 2D Gaussian
        gaussian_dist = torch.exp(((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        
        # Normalize to create probability distribution
        gaussian_prob = gaussian_dist / gaussian_dist.sum()
        _gaussian_prob_cache[cache_key] = gaussian_prob.flatten()
    
    return _gaussian_prob_cache[cache_key]


def sample_gaussian_and_zero_heightmap(heightmap: torch.Tensor, sigma: float, N: int) -> torch.Tensor:
    """
    Sample points in a flattened heightmap following a 2D Gaussian distribution centered in the middle,
    and set those sampled points to zero.
    
    Args:
        heightmap: Tensor of shape (num_envs, height*width) - flattened heightmap
        sigma: Standard deviation of the Gaussian distribution
        N: Number of points to sample following the Gaussian distribution
    
    Returns:
        Modified heightmap with N sampled points set to zero
    """
    num_envs, length = heightmap.shape
    h, w = 15, 10  # Fixed heightmap dimensions
    device = heightmap.device
    
    # Get cached Gaussian probability distribution
    flat_prob = _compute_gaussian_prob(h, w, sigma, device)
    
    # Sample N indices once (same for all environments)
    sampled_indices = torch.multinomial(flat_prob, N, replacement=True)
    
    # Apply the same sampling to all environments (vectorized)
    heightmap[:, sampled_indices] = 0.0
    
    return heightmap


def make_robot_roll_back_and_forth(
    joint_targets: torch.Tensor, sim_time: float, amplitude: float = 0.25, frequency_hz: float = 0.20
) -> torch.Tensor:
    """Oscillate roll around body x-axis (left-right lean)."""
    phase = np.sin(2.0 * np.pi * frequency_hz * sim_time)
    roll_offset = amplitude * phase

    # Left and right leg pairs move oppositely to induce roll motion.
    joint_targets[:, 4] += roll_offset  # FL_thigh_joint
    joint_targets[:, 6] += roll_offset  # RL_thigh_joint
    joint_targets[:, 5] -= roll_offset  # FR_thigh_joint
    joint_targets[:, 7] -= roll_offset  # RR_thigh_joint

    calf_comp = 0.5 * roll_offset
    joint_targets[:, 8] -= calf_comp  # FL_calf_joint
    joint_targets[:, 10] -= calf_comp  # RL_calf_joint
    joint_targets[:, 9] += calf_comp  # FR_calf_joint
    joint_targets[:, 11] += calf_comp  # RR_calf_joint
    return joint_targets

def compute_height_data(scene):
    # Get sensor/robot pose in world frame
    pos_w = scene["ray_caster"].data.pos_w          # (N, 3)
    quat_w = scene["ray_caster"].data.quat_w        # (N, 4) — w, x, y, z

    # Ray hit positions in world frame
    ray_hits_w = scene["ray_caster"].data.ray_hits_w  # (N, H, 3)
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
    height_data = -hits_in_base[..., 2] - 0.28 
    return height_data  

def compute_height_data2(scene):
    data = scene["ray_caster"].data
    pos_w      = data.pos_w        # (N, 3)
    quat_w     = data.quat_w       # (N, 4)  [w, x, y, z]
    ray_hits_w = data.ray_hits_w   # (N, H, 3)

    # ── translate ─────────────────────────────────────────────────────────────
    hits_rel = ray_hits_w - pos_w.unsqueeze(1)   # (N, H, 3)  — no copy, pure broadcast

    # ── rotate using inline quaternion sandwich — no reshape, no copy ─────────
    # q_inv = [w, -x, -y, -z]
    w  =  quat_w[:, 0:1].unsqueeze(1)   # (N, 1, 1)
    x  = -quat_w[:, 1:2].unsqueeze(1)   # (N, 1, 1)  ← negated for inverse
    y  = -quat_w[:, 2:3].unsqueeze(1)
    z  = -quat_w[:, 3:4].unsqueeze(1)

    vx = hits_rel[..., 0:1]   # (N, H, 1)
    vy = hits_rel[..., 1:2]
    vz = hits_rel[..., 2:3]

    # Standard quaternion-vector rotation: q ⊗ [0,v] ⊗ q*
    # Precompute cross product terms
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx

    # Rotated vector components
    rx = vx + 2.0 * (w * cx + y * vz - z * vy)
    ry = vy + 2.0 * (w * cy + z * vx - x * vz)
    rz = vz + 2.0 * (w * cz + x * vy - y * vx)

    # ── we only need Z ────────────────────────────────────────────────────────
    # Skip building the full (N, H, 3) tensor — compute only rz
    rz = vz + 2.0 * (w * (x * vy - y * vx) + x * (z * vx - x * vz) - y * (y * vz - z * vy))

    height_data = (-rz - 0.28).squeeze(-1)   # (N, H)
    return height_data

def make_robot_pitch_left_and_right(
    joint_targets: torch.Tensor, sim_time: float, amplitude: float = 0.35, frequency_hz: float = 0.25
) -> torch.Tensor:
    """Oscillate pitch around body y-axis (front-back tilt)."""
    phase = np.sin(2.0 * np.pi * frequency_hz * sim_time)
    pitch_offset = amplitude * phase

    # Front and rear leg pairs move oppositely to induce pitch motion.
    joint_targets[:, 4] += pitch_offset  # FL_thigh_joint
    joint_targets[:, 5] += pitch_offset  # FR_thigh_joint
    joint_targets[:, 6] -= pitch_offset  # RL_thigh_joint
    joint_targets[:, 7] -= pitch_offset  # RR_thigh_joint

    calf_comp = 0.5 * pitch_offset
    joint_targets[:, 8] -= calf_comp  # FL_calf_joint
    joint_targets[:, 9] -= calf_comp  # FR_calf_joint
    joint_targets[:, 10] += calf_comp  # RL_calf_joint
    joint_targets[:, 11] += calf_comp  # RR_calf_joint
    return joint_targets


def create_grid_from_cloud(scene):
    # Fast path: world -> lidar frame, bin to a flat heightmap (1D tensor).
    data = scene["ray_caster"].data
    ray_hits_w = data.ray_hits_w  # (num_envs, num_rays, 3)
    lidar_pos_w = data.pos_w      # (num_envs, 3)
    lidar_quat_w = data.quat_w    # (num_envs, 4)

    rays_rel_w = ray_hits_w - lidar_pos_w.unsqueeze(1)
    num_envs, num_rays, _ = rays_rel_w.shape
    rays_lidar = quat_apply(
        quat_conjugate(lidar_quat_w).unsqueeze(1).expand(num_envs, num_rays, 4).reshape(-1, 4),
        rays_rel_w.reshape(-1, 3),
    )

    cell_size_m = 0.1
    inv_cell_size = 1.0 / cell_size_m
    x_min, x_max = -MAX_RAY_DIST / 2.0, MAX_RAY_DIST
    y_min, y_max = -MAX_RAY_DIST / 2.0, MAX_RAY_DIST / 2.0
    x_cells = max(1, int(np.ceil((x_max - x_min) * inv_cell_size)))
    y_cells = max(1, int(np.ceil((y_max - y_min) * inv_cell_size)))
    num_cells = x_cells * y_cells

    valid = torch.isfinite(rays_lidar).all(dim=1)
    if not torch.any(valid):
        return torch.zeros(num_cells, device=ray_hits_w.device)

    pts = rays_lidar[valid]
    x_idx = torch.floor((pts[:, 0] - x_min) * inv_cell_size).long()
    y_idx = torch.floor((pts[:, 1] - y_min) * inv_cell_size).long()
    in_bounds = (x_idx >= 0) & (x_idx < x_cells) & (y_idx >= 0) & (y_idx < y_cells)
    if not torch.any(in_bounds):
        return torch.zeros(num_cells, device=ray_hits_w.device)

    flat_idx = x_idx[in_bounds] * y_cells + y_idx[in_bounds]
    z_vals = pts[in_bounds, 2]

    # Use -inf sentinel so empty cells can be zeroed with a single where.
    height_map_flat = torch.full((num_cells,), -torch.inf, device=ray_hits_w.device)
    height_map_flat.scatter_reduce_(0, flat_idx, z_vals, reduce="amax", include_self=True)
    height_map_flat = torch.where(torch.isfinite(height_map_flat), -height_map_flat, torch.zeros_like(height_map_flat))
    return height_map_flat.flip(0)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    triggered = True
    countdown = 42

    # Simulate physics
    while simulation_app.is_running():

        if count % 5000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            
            # root_state[:, 0] -= 5.8
            
            # root_state[:, 0] -= 3.8
            # root_state[:, 1] -= 3.0
            
            root_state[:, 0] -= ROOM_INNER_SIZE/2.0 +0.8
            root_state[:, 1] += 2.0*ROOM_CENTER_Y
            
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            target_add = torch.zeros_like(joint_pos)
            target_add[:,4] += 1.0
            target_add[:,5] += 1.0
            target_add[:,8] -= 1.0
            target_add[:,9] -= 1.0
            joint_pos += torch.rand_like(joint_pos) * 0.1 + target_add
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        #   - "FL_hip_joint"
        #   - "FR_hip_joint"
        #   - "RL_hip_joint"
        #   - "RR_hip_joint"
        #   - "FL_thigh_joint"
        #   - "FR_thigh_joint"
        #   - "RL_thigh_joint"
        #   - "RR_thigh_joint"
        #   - "FL_calf_joint"
        #   - "FR_calf_joint"
        #   - "RL_calf_joint"
        #   - "RR_calf_joint"

        targets = scene["robot"].data.default_joint_pos.clone()
        # targets = make_robot_roll_back_and_forth(targets, sim_time, amplitude=0.35)
        # targets = make_robot_pitch_left_and_right(targets, sim_time,amplitude=0.2)
        
        # print("pos: ", scene["robot"].data.default_joint_pos.clone())        
        scene["robot"].set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print("Robot base height (z):", scene["robot"].data.root_pos_w[:, 2])
        # print(scene["ray_caster"])
        height_data = create_grid_from_cloud(scene)
        cell_size_m = 0.1
        x_cells = max(1, int(np.ceil(((MAX_RAY_DIST) - (-MAX_RAY_DIST / 2.0)) / cell_size_m)))
        y_cells = max(1, int(np.ceil(((MAX_RAY_DIST / 2.0) - (-MAX_RAY_DIST / 2.0)) / cell_size_m)))
        height_data_2d = height_data.reshape(x_cells, y_cells)
        print(height_data_2d)
        # height_data = (
        #         scene["ray_caster"].data.pos_w[:, 2].unsqueeze(1) - scene["ray_caster"].data.ray_hits_w[..., 2] - 0.28
        #     )
        

        # sample_gaussian_and_zero_heightmap(heightmap=height_data, sigma=4.0, N=30)
        # height_data = height_data.reshape(args_cli.num_envs, 15, 10).flip(1,2)
        # height_data_half =  
        # height_data2 = (
        #         scene["ray_caster"].data.pos_w[:, 2].unsqueeze(1) - scene["ray_caster"].data.ray_hits_w[..., 2] - 0.28
        #     )[:, 10*n_rows_deleted:].reshape(args_cli.num_envs, 15, 1*10).flip(1,2)
        # print(height_data-height_data2)
        # x_data = (
                # scene["ray_caster"].data.pos_w[:, 0].unsqueeze(1) - scene["ray_caster"].data.ray_hits_w[..., 0] 
            # ).reshape(args_cli.num_envs, 15, 10).flip(1,2)# .clip(-1.0, 1.0)  
        # x_data_half = x_data[:, ::2, ::2] 
        # y_data = (
                # scene["ray_caster"].data.pos_w[:, 1].unsqueeze(1) - scene["ray_caster"].data.ray_hits_w[..., 1] 
            # )#.reshape(args_cli.num_envs, 15, 10).flip(1,2)# .clip(-1.0, 1.0)  
        # y_data_half = y_data[:, ::2, ::2] 
        # print(height_data)
        print("base height:", scene["robot"].data.root_pos_w[:, 2])
        # print(torch.sum(height_data == 0.0))
        # rays[torch.isinf(rays)] = 0
        
        
        
        # if not triggered:
        #     if countdown > 0:
        #         countdown -= 1
        #         continue
        #     data = scene["ray_caster"].data.ray_hits_w.cpu().numpy()
        #     np.save("cast_data.npy", data)
        #     triggered = True
        # else:
        #     continue


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = RaycasterSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
