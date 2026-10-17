"""Compatibility shims for upstream libraries that lag JAX.

Keep these in one place, each with the exact version evidence, so they can be
deleted the moment upstream catches up rather than lingering as folklore.
"""

from __future__ import annotations


def patch_device_put_replicated() -> bool:
    """Restore `jax.device_put_replicated`, which brax 0.14.2 still calls.

    brax 0.14.2 -- the newest release as of this writing -- calls
    jax.device_put_replicated at brax/training/agents/ppo/train.py:756. JAX
    removed it: accessing the attribute raises AttributeError from
    jax._src.deprecations, in 0.10.2 as well as 0.11.x. So no JAX pin fixes
    this; the function is gone and brax has not migrated.

    JAX's migration guide names jax.device_put as the drop-in. The original
    semantics: given a pytree and a device list, return the pytree with a new
    leading axis of size len(devices), one replica resident on each device.

    Returns True if a patch was installed, False if JAX still provides it
    (in which case the shim should be deleted).
    """
    import jax

    try:
        jax.device_put_replicated
        return False  # upstream restored it; this shim is now dead code
    except AttributeError:
        pass

    import jax.numpy as jnp
    import numpy as np

    def device_put_replicated(x, devices):
        n = len(devices)
        stacked = jax.tree_util.tree_map(
            lambda v: jnp.stack([jnp.asarray(v)] * n), x
        )
        if n == 1:
            # The common case here, and the one actually exercised: a single
            # GPU. Plain device_put is exact.
            return jax.device_put(stacked, devices[0])
        mesh = jax.sharding.Mesh(np.asarray(devices), ("d",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
        return jax.device_put(stacked, sharding)

    jax.device_put_replicated = device_put_replicated
    return True
