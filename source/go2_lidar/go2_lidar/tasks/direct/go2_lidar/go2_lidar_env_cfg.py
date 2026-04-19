# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumManager, CurriculumTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

#
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from .utils import terrain_levels_vel


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    CurriculumTermCfg(
        func=terrain_levels_vel,   # the standard manager-based term
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)
            # lin_vel_x=(0.5, 0.5), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(-3.14, 3.14)
        ),
    )
    
    

@configclass
class EventCfg:
    """Configuration for randomization."""

     # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class Go2LidarFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 60
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=512 * 1024,
            gpu_max_rigid_patch_count=9503130,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # command
    commands: CommandsCfg = CommandsCfg()

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    randomize = True

    # reward scales
    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -0.5
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.0e-4
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.01
    undesired_contact_reward_scale = -0.00
    flat_orientation_reward_scale = 0.0
    velocity_threshold = 0.3
    def_pos_reward_scale = -0.00
    stand_still_scale = 5.0


@configclass
class Go2LidarRoughEnvCfg(Go2LidarFlatEnvCfg):
    # env
    observation_space = 247
    
    curriculum: CurriculumCfg = CurriculumCfg()
    
    TERRAINS_CFG = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            #     proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
            # ),
            "pyramid_stairs_inv": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.05, 0.15),
                step_width=0.3,
                platform_width=2.0,
                border_width=1.0,
                holes=False,
            ),
            # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            #     proportion=0.4,
            #     step_height_range=(0.05, 0.15),
            #     step_width=0.3,
            #     platform_width=2.0,
            #     border_width=1.0,
            #     holes=False,
            # )
        },
    )
    # ROUGH_TERRAINS_CFG.num_cols = 2
    # ROUGH_TERRAINS_CFG.num_rows = 3
    # ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.1, 0.1)
    
    ROUGH_TERRAINS_CFG.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
    ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
    ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_step = 0.01
    
    
    # ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs"].proportion = 0.0
    # ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].proportion = 0.0
    # ROUGH_TERRAINS_CFG.sub_terrains["boxes"].proportion = 0.0
    # ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs_inv"].proportion = 1.0
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    #  # Heightmap configuration
    height_map_dist = 1.0
    # New grid-based heightmap config (values are in lidar frame).
    # NOTE: `res` is used as cell size in meters.
    res = 0.1
    x_range = [-0.5, 1.0]
    y_range = [-0.5, 0.5]
    # height_map_cells = int(2 * height_map_dist * res) ** 2  
    # observation_space = 53 + height_map_cells  
    
    # lidar_range = height_map_dist * 3.0 # * 1.4142135623730951  # sqrt(2)
    lidar_offset = (0.28945, 0.0, -0.04682)
    lidar_rotation = (0.13131596830945724, 0.0, 0.9913405653290647, 0.0)

    # New scanner path: multi-mesh ray-caster over /World.
    # height_scanner = MultiMeshRayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base",
    #     update_period=1 / 60,
    #     offset=MultiMeshRayCasterCfg.OffsetCfg(
    #         pos=lidar_offset,
    #         rot=lidar_rotation,
    #     ),
    #     mesh_prim_paths=["/World"],
    #     ray_alignment="base",
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=64, vertical_fov_range=[-0.0, 90.0], horizontal_fov_range=[-180, 180], horizontal_res=2.0
    #     ),
    #     max_distance=4.0,
    #     debug_vis=False,
    # )
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(
            pos=lidar_offset,
            rot=lidar_rotation,
        ),
        mesh_prim_paths=["/World"],
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=64, vertical_fov_range=[0.0, 90.0], horizontal_fov_range=[-180, 180], horizontal_res=2.0
        ),
        max_distance=4.0,
        debug_vis=False,
    )
   
    # height_scanner = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base",
    #     update_period=1 / 60,
    #     offset=RayCasterCfg.OffsetCfg(
    #         pos=lidar_offset,
    #         rot=lidar_rotation,
    #     ),
    #     mesh_prim_paths=["/World/ground"],
    #     ray_alignment="base",
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=128, vertical_fov_range=[-90.0, 90.0], horizontal_fov_range=[-180, 180], horizontal_res=2.0
    #     ),
    #     max_distance=lidar_range,
    #     debug_vis=False,
    # )

    # Pre-computed quaternion (w, x, y, z) from euler angles (-pi, pi - 2.8782, -pi)

    sigma = 4.00
    n_zeros = 30
    # the heightmap is 1.5 * 1, offseted by lidar offset + 0.25 on x such that it detects 1 metter in front of and 0.5 meters behind the lidar frame
    # on the real robot, from the lidar frame: grid 0.5 meters left and right and 1 meter front and 0.5 meters behind
    # Previous scanner path kept for reference (disabled):
    # height_scanner = RayCasterCfg(
    #     update_period=1 / 20,
    #     prim_path="/World/envs/env_.*/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.28945 + 0.25, 0.0, 0.5)),
    #     # ray_alignment="base",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.4, 0.9], ordering="yx"),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
   

    