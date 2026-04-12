import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional

from rsl_rl.networks import EmpiricalNormalization, MLP


class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_shape=(1, 15, 10),
        n_layers=3,
        channels=(16, 32, 64),
        kernel_sizes=(3, 3, 3),
        strides=(1, 2, 2),
        paddings=(1, 1, 1),
        output_layer=128,
        activation=nn.ReLU,
    ):
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (C, H, W), got: {input_shape}")
        if not (len(channels) == len(kernel_sizes) == len(strides) == len(paddings) == n_layers):
            raise ValueError(
                "Expected lengths of channels/kernel_sizes/strides/paddings to match n_layers. "
                f"Got n_layers={n_layers}, channels={len(channels)}, kernel_sizes={len(kernel_sizes)}, "
                f"strides={len(strides)}, paddings={len(paddings)}"
            )

        in_channels = input_shape[0]
        conv_layers = []
        for i in range(n_layers):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            )
            conv_layers.append(activation())
            in_channels = channels[i]

        self.conv = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy)
            conv_flat_dim = conv_out.flatten(start_dim=1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_flat_dim, output_layer),
            activation(),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class ActorCriticCNN(nn.Module):
    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(256, 256, 128),
        critic_hidden_dims=(256, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        cnn_n_layers=3,
        cnn_channels=(16, 32, 64),
        cnn_kernel_sizes=(3, 3, 3),
        cnn_strides=(1, 2, 2),
        cnn_paddings=(1, 1, 1),
        cnn_output_layer=128,
        proprio_hidden_dim=64,
        grid_shape=(1, 15, 10),
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticCNN.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.noise_std_type = noise_std_type
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization

        # Distillation policy observations are expected as obs["policy"] = {"proprio": ..., "grid": ...}.
        policy_group_name = obs_groups["policy"][0]
        sample_group = obs[policy_group_name]
        sample_grid = sample_group["grid"]
        sample_proprio = sample_group["proprio"]

        inferred_grid_shape = tuple(sample_grid.shape[1:])
        inferred_proprio_dim = sample_proprio.shape[-1]

        if tuple(grid_shape) != inferred_grid_shape:
            grid_shape = inferred_grid_shape

        self.cnn = CNNEncoder(
            input_shape=grid_shape,
            n_layers=cnn_n_layers,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            paddings=cnn_paddings,
            output_layer=cnn_output_layer,
        )

        self.proprio_mlp = nn.Sequential(
            nn.Linear(inferred_proprio_dim, proprio_hidden_dim),
            nn.ReLU(),
        )

        fused_dim = cnn_output_layer + proprio_hidden_dim
        self.actor = MLP(fused_dim, num_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(fused_dim, 1, list(critic_hidden_dims), activation)

        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(inferred_proprio_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(inferred_proprio_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution: Optional[Normal] = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized. Call act() first.")
        return self.distribution.mean

    @property
    def action_std(self):
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized. Call act() first.")
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized. Call act() first.")
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def _fuse_group_obs(self, group_obs, normalize_proprio=None):
        proprio = group_obs["proprio"]
        grid = group_obs["grid"]

        if normalize_proprio is not None:
            proprio = normalize_proprio(proprio)

        grid_feat = self.cnn(grid)
        prop_feat = self.proprio_mlp(proprio)
        return torch.cat([grid_feat, prop_feat], dim=-1)

    def get_actor_features(self, obs):
        policy_group_name = self.obs_groups["policy"][0]
        return self._fuse_group_obs(obs[policy_group_name], normalize_proprio=self.actor_obs_normalizer)

    def get_critic_features(self, obs):
        critic_group_name = self.obs_groups["critic"][0]
        return self._fuse_group_obs(obs[critic_group_name], normalize_proprio=self.critic_obs_normalizer)

    def update_distribution(self, obs):
        mean = self.actor(self.get_actor_features(obs))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        if self.distribution is None:
            raise RuntimeError("Action distribution failed to initialize.")
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(self.get_actor_features(obs))

    def evaluate(self, obs, **kwargs):
        return self.critic(self.get_critic_features(obs))

    def get_actions_log_prob(self, actions):
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized. Call act() first.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(obs[self.obs_groups["policy"][0]]["proprio"])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(obs[self.obs_groups["critic"][0]]["proprio"])

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def forward(self, obs):
        return self.act_inference(obs), self.evaluate(obs)

