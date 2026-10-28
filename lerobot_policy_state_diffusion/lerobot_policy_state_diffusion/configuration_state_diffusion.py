from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("state_diffusion")
@dataclass
class StateDiffusionConfig(DiffusionConfig):
    def validate_features(self) -> None:
        if not self.image_features and self.env_state_feature is None:
            if self.robot_state_feature is None:
                raise ValueError("state_diffusion needs observation.state among the inputs.")
            return
        super().validate_features()
