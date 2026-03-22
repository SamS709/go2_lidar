#!/usr/bin/env python3

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
import torch

def quat_rotate_inverse(q, v):
    """
    Rotate vector v by the inverse of quaternion q.
    q: quaternion [w, x, y, z]
    v: vector [x, y, z]
    Returns rotated vector
    """
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    q_conj = torch.tensor([q_w, -q_x, -q_y, -q_z])
    t = 2.0 * torch.cross(q_conj[1:], v)
    return v + q_conj[0] * t + torch.cross(q_conj[1:], t)

"""
self._robot.data.root_ang_vel_b,
self._robot.data.projected_gravity_b,
self._commands.get_command("base_velocity"),
self._robot.data.joint_pos - self._robot.data.default_joint_pos,
self._robot.data.joint_vel,
height_data,
self._actions,
"""

def get_obs(
    highstate_msg: SportModeState_,
    lowstate_msg: LowState_,
    vel_x: float,
    vel_y: float,
    vel_yaw: float,
    prev_actions: torch.tensor,
    mapper,
):
    """
    Extract observations from LowState message only (no SportModeState).
    
    Args:
        lowstate_msg: LowState message from robot
        vel_x: Forward velocity command (m/s)
        vel_y: Lateral velocity command (m/s)
        vel_yaw: Yaw rate command (rad/s)
        height: Height command (m)
        prev_actions: Previous actions in policy order (12,)
        mapper: Mapper object for joint remapping
    
    Returns:
        obs: numpy array of shape (48,) with observation vector
        
    Observation structure (48 dimensions):
    - obs[0:3]   : Base angular velocity (from IMU) 
    - obs[3:6]   : Gravity direction (from IMU)
    - obs[6:9]   : Command velocity (x, y, yaw)
    - obs[9]     : Height command
    - obs[10:22] : Joint positions relative to default (12 joints)
    - obs[22:34] : Joint velocities (12 joints)
    - obs[34:46] : Previous actions (12 values)
    - obs[46:50] : Feet contact signal
    """
    
    
    # MAPPING ROBOT -> POLICY
    motor_states = lowstate_msg.motor_state[:12]
    
    current_joint_pos_sdk = torch.tensor([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = torch.tensor([motor_states[i].dq for i in range(12)])
    
    current_joint_pos_policy = mapper.remap_joints_by_name(
        current_joint_pos_sdk, mapper.target_names, mapper.source_names, mapper.target_to_source
    )
    current_joint_vel_policy = mapper.remap_joints_by_name(
        current_joint_vel_sdk, mapper.target_names, mapper.source_names, mapper.target_to_source
    )
    default_pos_policy = mapper.default_pos_policy

    # FILLING OBS VECTOR
    obs = torch.zeros(195)

    # Base vel
    
    # Base angular velocity (gyroscope) (obs[0:3])
    obs[0:3] = torch.tensor([
        lowstate_msg.imu_state.gyroscope[0],
        lowstate_msg.imu_state.gyroscope[1],
        lowstate_msg.imu_state.gyroscope[2]
    ])
    
    # Computing projected gravity from IMU sensor
    quat = torch.tensor([
        lowstate_msg.imu_state.quaternion[0],  # w
        lowstate_msg.imu_state.quaternion[1],  # x
        lowstate_msg.imu_state.quaternion[2],  # y
        lowstate_msg.imu_state.quaternion[3]   # z
    ])
    
    gravity_world = torch.tensor([0.0, 0.0, -1.0])
    gravity_b = quat_rotate_inverse(quat, gravity_world)
    obs[3:6] = gravity_b
    
    # Command velocity (obs[6:9])
    obs[6:9] = torch.tensor([vel_x, vel_y, vel_yaw])
    
    
    # Fill joint positions (obs[10:22]) in policy order
    obs[9:21] = current_joint_pos_policy - default_pos_policy
    
    # Fill joint velocities (obs[22:34]) in policy order
    obs[21:33] = current_joint_vel_policy
    
    idx = 33+150
    num_zeros = 30
    random_indices = torch.randperm(150)[:num_zeros]
    height_map = torch.zeros(150, dtype = torch.float32) + highstate_msg.position[2] - 0.28 + (2.0 * torch.rand_like(torch.zeros(150)) - 1.0) * float(0.1)
    height_map[random_indices] = 0
    obs[33:idx] = height_map
    # Set 3 random positions to zero
    
    # print(obs[36:idx].reshape(15,10))

    # Previous actions (obs[34:46])
    obs[idx:idx + 12] = prev_actions
    
    # print(obs[46:50])
    
    return obs
