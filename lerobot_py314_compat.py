"""Python 3.14 compatibility shim for draccus (used by lerobot config loading).

Import this module **before** anything that triggers ``draccus.parse`` — i.e. before
``DiffusionPolicy.from_pretrained`` / ``PreTrainedConfig.from_pretrained`` and before the
``lerobot-train`` CLI.

Why this is needed
------------------
This box runs Python 3.14 (required so ``rclpy`` for ROS 2 Lyrical and ``torch`` can be
imported in one interpreter — see docs/LYRICAL_KINOVA.md). lerobot 0.5.0 loads every
policy/train config through ``draccus.parse(cls, config_file, args=[])``, which builds an
``argparse`` parser even when reading purely from a file.

draccus 0.8.0 passes the field's *type annotation* straight to ``add_argument(type=...)``.
For ``Optional[X]`` / ``X | None`` fields (e.g. ``root: str | None``,
``input_features: Dict[str, PolicyFeature] | None``) that annotation is a non-callable
union object. On Python <= 3.13 ``argparse`` only invokes ``type`` when it actually parses
a string, so with ``args=[]`` it was never called and everything worked. Python 3.14's
``argparse.add_argument`` validates callability *eagerly*, so it raises
``TypeError: X | None is not callable`` at parser-construction time — breaking both the
train CLI and ``from_pretrained``.

The fix
-------
Wrap ``draccus.utils.canonicalize_union`` so a non-callable (Optional/union) result is
replaced by a callable. draccus only ever *invokes* this ``type`` when parsing a CLI
*string* argument; the file-based loading path (``args=[]``) never calls it, so a callable
placeholder preserves decoding behavior exactly while satisfying Python 3.14's check.
"""
import typing

import draccus.utils as _draccus_utils

_orig_canonicalize_union = _draccus_utils.canonicalize_union


def _callable_canonicalize_union(t):
    r = _orig_canonicalize_union(t)
    if callable(r):
        return r
    # Optional[X] with a single callable non-None arg -> use X (correct for CLI too).
    non_none = [a for a in typing.get_args(r) if a is not type(None)]
    if len(non_none) == 1 and callable(non_none[0]):
        return non_none[0]
    # e.g. Dict[str, PolicyFeature] | None: no simple callable. Use a harmless
    # passthrough — never invoked on the args=[] file-loading path lerobot uses.
    return str


def apply():
    """Idempotently install the patch."""
    if getattr(_draccus_utils.canonicalize_union, "_py314_patched", False):
        return
    _callable_canonicalize_union._py314_patched = True
    _draccus_utils.canonicalize_union = _callable_canonicalize_union


apply()
