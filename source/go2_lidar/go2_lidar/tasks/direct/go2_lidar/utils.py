
from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


from collections.abc import Sequence


def terrain_levels_vel(
    env, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    # Randomly update half of the reset environments, and use the distance-based
    # curriculum for the remaining environments.
    num_random = distance.numel() // 2
    random_order = torch.randperm(distance.numel(), device=distance.device)
    random_ids = random_order[:num_random]
    formula_ids = random_order[num_random:]

    move_up = torch.zeros(distance.numel(), dtype=torch.bool, device=distance.device)
    move_down = torch.zeros_like(move_up)
    random_move_up = torch.rand(num_random, device=distance.device) < 0.5
    move_up[random_ids] = random_move_up
    move_down[random_ids] = ~random_move_up

    # Formula-based curriculum for the remaining environments.
    formula_distance = distance[formula_ids]
    formula_env_ids = env_ids[formula_ids]
    formula_command = command[formula_env_ids, :2]
    formula_move_up = formula_distance > terrain.cfg.terrain_generator.size[0] / 2
    formula_move_down = formula_distance < torch.norm(
        formula_command, dim=1
    ) * env.max_episode_length_s * 0.5
    formula_move_down &= ~formula_move_up
    move_up[formula_ids] = formula_move_up
    move_down[formula_ids] = formula_move_down

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    # print("coucou")
    return torch.mean(terrain.terrain_levels.float())