"""Isaac-free perception stack for the Go2 navigation task.

Everything in this package depends on torch only (plus ``rsl_rl`` for the odometry GRU), so the exact same code can
run in the simulator and on the robot.

    LocomotionController  - frozen low-level policy (TorchScript) + its observation/action bookkeeping
    HeightGrid            - ray hits (lidar frame) -> 2D height grid (policy convention)
    OdometryEstimator     - GRU velocity/attitude network + dead-reckoning + pose histories
    MapPerception         - rolling Kalman elevation map + scan-to-map matching + ego-crop denoiser
"""

from .height_grid import HeightGrid, rays_in_lidar_frame
from .locomotion import LocomotionController, build_loc_obs, build_proprio


__all__ = [
    "HeightGrid", "rays_in_lidar_frame", "LocomotionController", "build_loc_obs", "build_proprio"
]