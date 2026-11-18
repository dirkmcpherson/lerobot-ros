"""Runtime fixes for lerobot 0.6.1 that the HIL-SERL entry points need. Nothing in the
installed lerobot package is edited; each fix is applied in memory at start-up.

1. ``ResetConfig.fixed_reset_joint_positions`` is annotated ``Any | None``. draccus 0.11.6
   (the version lerobot pins) cannot encode a dataclass with such a field: it calls
   ``isinstance(value, typing.Any)``, which raises. ``cfg.to_dict()`` therefore fails for
   every ``gym_manipulator`` env config whose ``processor.reset`` is set, and the learner
   calls it on its first lines (and again when saving a checkpoint). The field's type is
   narrowed to ``list[float] | None`` here, which is what it holds.

2. The learner builds its offline (demo) buffer from ``make_dataset``, which in 0.6.1
   returns images as ``uint8`` in [0, 255] (``return_uint8=True``). ``ReplayBuffer``
   stores them as float32 without rescaling, while the actor sends images as float in
   [0, 1]. The demo images were therefore 255 times brighter than the online ones in
   every mixed batch, and the learner's checkpoint save crashed on them (the image
   writer rejects floats above 1). ``_lerobotdataset_to_transitions`` is wrapped here so
   uint8 tensors in a transition's state come out as float in [0, 1].
"""

import dataclasses


def apply() -> None:
    from lerobot.envs.configs import ResetConfig

    fixed = list[float] | None
    for field in dataclasses.fields(ResetConfig):
        if field.name == "fixed_reset_joint_positions":
            field.type = fixed
    ResetConfig.__annotations__["fixed_reset_joint_positions"] = fixed

    _scale_offline_images()


def uint8_to_unit_float(state: dict | None) -> dict | None:
    """A copy of ``state`` with every uint8 tensor as float32 in [0, 1]."""
    import torch

    if state is None:
        return None
    return {k: v.to(torch.float32) / 255.0 if isinstance(v, torch.Tensor) and v.dtype == torch.uint8 else v
            for k, v in state.items()}


def _scale_offline_images() -> None:
    from lerobot.rl.buffer import ReplayBuffer

    original = ReplayBuffer.__dict__["_lerobotdataset_to_transitions"].__func__
    if getattr(original, "_gen3_lite_scaled", False):
        return

    def to_transitions(dataset, state_keys=None):
        out = []
        for t in original(dataset=dataset, state_keys=state_keys):
            t = dict(t)
            t["state"] = uint8_to_unit_float(t["state"])
            t["next_state"] = uint8_to_unit_float(t["next_state"])
            out.append(t)
        return out

    to_transitions._gen3_lite_scaled = True
    ReplayBuffer._lerobotdataset_to_transitions = staticmethod(to_transitions)
