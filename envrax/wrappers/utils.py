# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import chex
import jax
import jax.numpy as jnp

_LUMA = jnp.array([0.299, 0.587, 0.114], dtype=jnp.float32)


def to_gray(obs: chex.Array) -> chex.Array:
    """Apply NTSC luminance weights: `uint8[H, W, 3]` → `uint8[H, W]`."""
    return jnp.dot(obs.astype(jnp.float32), _LUMA).astype(jnp.uint8)


def resize(obs: chex.Array, out_h: int, out_w: int) -> chex.Array:
    """Bilinear-resize `uint8[H, W]` → `uint8[out_h, out_w]`."""
    resized = jax.image.resize(
        obs.astype(jnp.float32), (out_h, out_w), method="bilinear"
    )
    return resized.astype(jnp.uint8)
