
from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


from collections.abc import Sequence

def terrain_levels_vel(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    terrain = env._terrain                      # your own terrain object (old _setup_scene handling)
    cmd_path = env._cmd_path[env_ids]
    ratio = env._prog_path[env_ids] / cmd_path.clamp(min=1e-6)

    died = env.reset_terminated[env_ids]
    timed_out = env.reset_time_outs[env_ids] & ~died
    enough = cmd_path > 2.0                     # ignore standing / barely-commanded episodes

    move_up = timed_out & enough & (ratio > 0.75)
    move_down = died | (enough & (ratio < 0.4))
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    env.extras.setdefault("log", {})
    env.extras["log"]["Curriculum/frac_up"] = move_up.float().mean()
    env.extras["log"]["Curriculum/frac_down"] = move_down.float().mean()
    env.extras["log"]["Curriculum/tracking_ratio"] = ratio[enough].mean() if enough.any() else torch.zeros((), device=ratio.device)
    return torch.mean(terrain.terrain_levels.float())