"""``--policy.type=state_diffusion``: lerobot's Diffusion Policy, conditioned on joint state alone.

lerobot's ``DiffusionConfig.validate_features`` insists on at least one image or an
``observation.environment_state``, but the model itself conditions on
``observation.state`` and treats both of those as optional. This plugin only drops
that check, for robots recorded with joint state and no cameras. Such a policy
cannot see the scene: it can only learn motions that do not depend on where
objects are (e.g. a cup always placed at the same spot).
"""

from .configuration_state_diffusion import StateDiffusionConfig
from .modeling_state_diffusion import StateDiffusionPolicy
from .processor_state_diffusion import make_state_diffusion_pre_post_processors
