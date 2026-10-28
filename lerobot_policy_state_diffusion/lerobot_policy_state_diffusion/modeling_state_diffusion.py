from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_state_diffusion import StateDiffusionConfig


class StateDiffusionPolicy(DiffusionPolicy):
    config_class = StateDiffusionConfig
    name = "state_diffusion"

    def forward(self, batch):
        # DiffusionModel.compute_loss asserts an image or environment-state key is
        # present. With no env_state_feature configured the conditioning never reads
        # that key, so an empty placeholder satisfies the assert without changing anything.
        if OBS_IMAGES not in batch and OBS_ENV_STATE not in batch:
            batch = {**batch, OBS_ENV_STATE: batch[OBS_STATE][..., :0]}
        return super().forward(batch)
