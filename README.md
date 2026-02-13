[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.1-silver.svg)](https://isaac-sim.github.io/IsaacLab/)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Lidar integration [BETA]

## Overview

This project is a part of the go2_lidar project which aims to make a Sim2Real for Unitree Go2 quadruped locomotion

The goal is to add the lidar of the go2 as a perception module so that the robot can walk on rough environments.

**Key Features:**

1) [**Training**](#1-training)
    - [`a) Train`](#a-train) a policy for go2 robot using direct based environnement. The policy follows the commands sent by the user: linear (x/y) velocitiezs // angular (z) velocity // base height.
    - [`b) Test`](#b-test) it using keyboard in Isaacsim.
2) [**Sim2Sim**](#2-sim2sim)
    - [`a) Newton`](#a-newton) from PhysX to Newton using Newton branch of Isaaclab repo.
    - [`b) Unitree_mujoco`](#b-unitree_mujoco) from PhysX to Mujoco using the unitree_mujoco repo.
    - [`c) Huro`](#c-huro) sim2sim in huro environment (github of a researcher at LORIA).
3) [**Sim2Real**](#3-sim2real)
    - [`a) Unitree_python_sdk2`](#a-unitree_python_sdk2) sim2real in unitree_python_sdk2 using proprietary dds developed by unitree.
    - [`b) Huro`](#b-huro) sim2real in huro using ros2.


## Installation

- Install **Isaac Lab** following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) (tested with **Isaac Sim 5.1.0** and **Isaac Lab v2.3.1**).

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
  
    ```bash
    git clone https://github.com/SamS709/go2_lidar.git
    ```
  
- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    cd go2_isaaclab
    python -m pip install -e source/go2_lidar
    ```

## 1) Training

### a) Train

Make sure you are in your the classic Isaac Lab Python environment (not the Newton branch).

- Train the Go2 locomotion environment:

    ```bash
    cd go2_lidar
    python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-Lidar-Direct-v0 --num_envs 4096 --headless
    ```

### b) Test

- Run the trained policy :

    ```bash
    python scripts/rsl_rl/play.py --task Isaac-Velocity-Go2-Asymmetric-v0 --num_envs 8 
    ```

- Control the robot with the keyboard (here, a pretrained checkpoint is used for convenience):

    ```bash
    python scripts/control/go2_locomotion.py --checkpoint pretrained_checkpoint/pretrained_checkpoint.pt --visualize
    ```

    <img src="images/commands_control.png" width="400"/>

    Controls:

  - **Up/Down arrows**: Increase/decrease the robot's forward/backward velocity (x-axis)
  - **Left/Right arrows**: Increase/decrease the robot's left/right velocity (y-axis)
  - **F/G keys**: Increase/decrease the robot's angular velocity (yaw rotation)

## 2) Sim2Sim



