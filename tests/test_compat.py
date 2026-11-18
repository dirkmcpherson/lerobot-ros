"""compat.apply(): the offline (demo) replay buffer holds images in [0, 1], like the online one."""

import torch


class _TinyDataset:
    """What ReplayBuffer.from_lerobot_dataset reads from a LeRobotDataset built by make_dataset:
    len(), indexing, and images as uint8 in [0, 255] (make_dataset passes return_uint8=True)."""

    def __init__(self, n=4):
        self.frames = []
        for i in range(n):
            img = torch.full((3, 8, 8), 255 if i % 2 else 51, dtype=torch.uint8)
            self.frames.append({
                "observation.images.front": img,
                "observation.state": torch.full((10,), float(i)),
                "action": torch.zeros(4),
                "next.reward": torch.tensor(0.0),
                "next.done": torch.tensor(i == n - 1),
                "episode_index": torch.tensor(0),
            })

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]


def test_offline_buffer_images_are_unit_float():
    from hilserl_gen3_lite import compat

    compat.apply()
    compat.apply()  # idempotent: must not wrap twice (a second /255 would darken the demos)
    from lerobot.rl.buffer import ReplayBuffer

    keys = ["observation.images.front", "observation.state"]
    buf = ReplayBuffer.from_lerobot_dataset(_TinyDataset(), device="cpu", state_keys=keys, storage_device="cpu")

    img = buf.states["observation.images.front"][: len(buf)]
    assert img.dtype == torch.float32
    assert float(img.max()) == 1.0
    assert abs(float(img.min()) - 0.2) < 1e-6
    nxt = buf.next_states["observation.images.front"][: len(buf)] if hasattr(buf, "next_states") else img
    assert float(nxt.max()) <= 1.0
    # Non-image state is untouched.
    assert torch.equal(buf.states["observation.state"][2], torch.full((10,), 2.0))


def test_uint8_to_unit_float_leaves_floats_alone():
    from hilserl_gen3_lite.compat import uint8_to_unit_float

    out = uint8_to_unit_float({"a": torch.tensor([255], dtype=torch.uint8), "b": torch.tensor([3.0])})
    assert out["a"].dtype == torch.float32 and float(out["a"]) == 1.0
    assert float(out["b"]) == 3.0
    assert uint8_to_unit_float(None) is None
