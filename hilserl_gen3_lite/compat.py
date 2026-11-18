"""Runtime fixes for lerobot 0.6.1 that the HIL-SERL entry points need. Nothing in the
installed lerobot package is edited; each fix is applied in memory at start-up.

1. ``ResetConfig.fixed_reset_joint_positions`` is annotated ``Any | None``. draccus 0.11.6
   (the version lerobot pins) cannot encode a dataclass with such a field: it calls
   ``isinstance(value, typing.Any)``, which raises. ``cfg.to_dict()`` therefore fails for
   every ``gym_manipulator`` env config whose ``processor.reset`` is set, and the learner
   calls it on its first lines (and again when saving a checkpoint). The field's type is
   narrowed to ``list[float] | None`` here, which is what it holds.
"""

import dataclasses


def apply() -> None:
    from lerobot.envs.configs import ResetConfig

    fixed = list[float] | None
    for field in dataclasses.fields(ResetConfig):
        if field.name == "fixed_reset_joint_positions":
            field.type = fixed
    ResetConfig.__annotations__["fixed_reset_joint_positions"] = fixed
