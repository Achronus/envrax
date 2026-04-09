import chex
import jax
import jax.numpy as jnp

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
