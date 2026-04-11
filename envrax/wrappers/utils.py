from typing import TYPE_CHECKING, Any

import chex
import jax
import jax.numpy as jnp

from envrax.spaces import Box

if TYPE_CHECKING:
    from envrax.base import JaxEnv

_LUMA = jnp.array([0.299, 0.587, 0.114], dtype=jnp.float32)


def to_gray(obs: chex.Array) -> chex.Array:
    """Apply NTSC luminance weights: `uint8[H, W, 3]` → `uint8[H, W]`."""
    return jnp.dot(obs.astype(jnp.float32), _LUMA).astype(jnp.uint8)


def resize(obs: chex.Array, out_h: int, out_w: int) -> chex.Array:
    """Bilinear-resize `uint8[H, W]` or `uint8[H, W, C]` to `(out_h, out_w[, C])`."""
    if obs.ndim == 3:
        shape = (out_h, out_w, obs.shape[-1])
    else:
        shape = (out_h, out_w)
    resized = jax.image.resize(obs.astype(jnp.float32), shape, method="bilinear")
    return resized.astype(jnp.uint8)


def require_box(
    env: "JaxEnv",
    wrapper_name: str,
    *,
    rank: int | tuple[int, ...] | None = None,
    last_dim: int | None = None,
    dtype: Any = None,
) -> Box:
    """
    Validate that `env.observation_space` is a `Box` matching the given constraints.

    Used by image-processing wrappers to fail fast at construction time
    instead of producing cryptic errors on the first `reset`.

    Parameters
    ----------
    env : JaxEnv
        Environment whose observation space is being validated
    wrapper_name : str
        Name of the wrapper performing the check (used in error messages)
    rank : int | tuple[int, ...] (optional)
        Required rank of the observation. A tuple permits multiple ranks
    last_dim : int (optional)
        Required size of the trailing dimension (e.g. `3` for RGB)
    dtype : Any (optional)
        Required element dtype (e.g. `jnp.uint8`)

    Returns
    -------
    space : Box
        The validated observation space

    Raises
    ------
    TypeError
        If the observation space is not a `Box`
    ValueError
        If any rank/last_dim/dtype constraint fails
    """
    space = env.observation_space
    if not isinstance(space, Box):
        raise TypeError(
            f"{wrapper_name} requires a Box observation space, "
            f"got {type(space).__name__}"
        )
    if rank is not None:
        allowed = (rank,) if isinstance(rank, int) else rank
        if len(space.shape) not in allowed:
            raise ValueError(
                f"{wrapper_name} requires observation rank in {allowed}, "
                f"got shape {space.shape}"
            )
    if last_dim is not None and space.shape[-1] != last_dim:
        raise ValueError(
            f"{wrapper_name} requires last dim = {last_dim}, got shape {space.shape}"
        )
    if dtype is not None and space.dtype != dtype:
        raise ValueError(
            f"{wrapper_name} requires {jnp.dtype(dtype).name} dtype, "
            f"got {jnp.dtype(space.dtype).name}"
        )
    return space
