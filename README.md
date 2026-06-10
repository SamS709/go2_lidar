[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.1-silver.svg)](https://isaac-sim.github.io/IsaacLab/)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Lidar integration [BETA]

<img src="/images/gazebo_lidar_rl_compressed.gif" alt="Gazebo lidar RL demo" width="400" />
<img src="/images/sim2real_rough2.png" alt="Sim to real deployement" width="400" />

## Overview

<p style="font-size: 1.2rem;"> This project is a part of the <a href="https://github.com/SamS709/go2_isaaclab">go2_isaaclab</a> project which aims to make a Sim2Real for Unitree Go2 quadruped locomotion </p>

The goal of this repo is to add the lidar of the go2 as a perception module so that the robot can walk on rough environments. This is done by adding the observations of the 3D lidar that comes with the robot.

**Key Features:**

1) [**Training**](#1-training)
    - A policy for go2 robot using direct based environnement. The policy follows the commands sent by the user: linear (x/y) velocitiezs // angular (z) velocity // base height.
2) [**Sim2Sim**](#2-sim2sim)
    - Sim2Sim in Huro environment (GitHub of the HUCEBOT team at LORIA).
3) [**Sim2Real**](#3-sim2real)
    - Sim2Real in Huro using ros2.
4) [**How does it work ?**](#4-howDoesItWork)
    - Sim2Real in Huro using ros2.


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

To see how the lidar observations are computed, go to [lidar_info.md](lidar_info.md).

### a) Train

Make sure you are in your the classic Isaac Lab Python environment (not the Newton branch).

- Train the Go2 locomotion environment:

    ```bash
    cd go2_lidar
    python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-CNN-Lidar-Direct-v0 --num_envs 4096 --headless
    ```

### b) Test

- Run the trained policy :

    ```bash
    python scripts/rsl_rl/play.py --task Isaac-Velocity-Rough-Go2-CNN-Lidar-Direct-v0 --num_envs 512
    ```

<img src="/images/robot_rough_vis_cmd.png" alt="Sim to real deployement" width="800" />

## 2) Sim2Sim

Using HURO repository from the HUCEBOT team at INRIA.
Simulated in gazebo.

See the instructions given [here](https://github.com/hucebot/huro/tree/sami).

The result without navigation (velocities sent with joystick):

The result with navigation for the moment: 

<img src="/images/gazebo_lidar_rl_compressed.gif" alt="Gazebo lidar RL demo" width="800" />

## 3) Sim2Real

Using HURO repository from the HUCEBOT team at INRIA.

See the instructions given [here](https://github.com/hucebot/huro/tree/sami).

<img src="/images/sim2real_rough2.png" alt="Sim to real deployement" width="800" />


## 4)  How does it work ?

See the used rewards at To see how the lidar observations are computed, go to [go2_lidar_env.py](source/go2_lidar/go2_lidar/tasks/direct/go2_lidar/go2_lidar_env.py). 

Here is the choosen structure for the neural net (A benchmarsk has been done resulting in the following classification: MLP < RNN< CNN)

<img src="/images/neural_net.png" alt="Neural network" width="800" />

