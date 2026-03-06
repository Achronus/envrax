# Envrax

Core JAX-native RL environment framework — base classes, spaces, wrappers, and a shared
registry. Every environment in the [Envrax suite](#the-envrax-suite) builds on this package.

All environment logic follows a **stateless functional design**: state is an explicit
`chex.dataclass` pytree passed to and returned from every call, making the full
`reset → step → rollout` pipeline compatible with `jax.jit`, `jax.vmap`, and
`jax.lax.scan` with zero modification.

## Features

- **`JaxEnv` base class** — standardised `reset(rng, params)` / `step(rng, state, action, params)` / `step_env(...)` interface every suite environment implements.
- **`EnvState` + `EnvParams`** — `chex.dataclass` pytrees for state and static config; fully composable with `jax.tree_util`, `optax`, and `flax`.
- **`Discrete` + `Box` spaces** — typed observation and action space definitions with `sample()` and `contains()`.
- **`VmapEnv`** — wraps any `JaxEnv` to run `N` parallel instances via `jax.vmap`. No changes to the underlying environment needed.
- **`JitWrapper`** — wraps any `JaxEnv` to compile `reset` and `step` with `jax.jit` on construction and cache compiled kernels to disk.
- **`make()` factory** — create any registered environment with optional wrappers and automatic JIT compilation in one call.
- **Composable wrappers** — ten generic preprocessing wrappers covering observation transforms, reward shaping, and episode tracking; all compatible with `jit`, `vmap`, and `lax.scan`.
- **Shared registry** — `register()` / `make()` let any installed suite package expose its environments through a single `envrax.make("name-v0")` call.

## Requirements

- Python 3.13+
- JAX 0.9+ (CPU, CUDA, or TPU backend)

## Installation

```bash
pip install envrax
```

Or from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Achronus/envrax
cd envrax
uv sync
```

## Quick Start

### Implementing a `JaxEnv`

```python
import chex
import jax
import jax.numpy as jnp

from envrax import JaxEnv, EnvState, EnvParams
from envrax.spaces import Box, Discrete


@chex.dataclass
class BallState(EnvState):
    ball_x: jnp.float32
    ball_y: jnp.float32


class BallEnv(JaxEnv):
    @property
    def observation_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=4)

    def reset(self, rng: chex.PRNGKey, params: EnvParams):
        rng_x, rng_y = jax.random.split(rng)
        state = BallState(
            step=jnp.int32(0),
            done=jnp.bool_(False),
            ball_x=jax.random.uniform(rng_x),
            ball_y=jax.random.uniform(rng_y),
        )
        obs = jnp.array([state.ball_x, state.ball_y])
        return obs, state

    def step(self, rng: chex.PRNGKey, state: BallState, action: chex.Array, params: EnvParams):
        new_state = state.replace(step=state.step + 1)
        obs = jnp.array([new_state.ball_x, new_state.ball_y])
        reward = jnp.float32(1.0)
        done = new_state.step >= params.max_steps
        return obs, new_state.replace(done=done), reward, done, {}
```

### `step_env()` — auto-reset on episode end

`JaxEnv.step_env()` wraps `step()` to transparently reset the environment when `done`
is `True`, returning the first observation of the new episode. This is what `VmapEnv`
uses internally, so each parallel instance resets independently.

```python
rng = jax.random.PRNGKey(0)
params = EnvParams(max_steps=100)
env = BallEnv()

obs, state = env.reset(rng, params)
obs, state, reward, done, info = env.step_env(rng, state, action=jnp.int32(0), params=params)
```

### `VmapEnv` — parallel environments

```python
from envrax.wrappers import VmapEnv

rng = jax.random.PRNGKey(0)
params = EnvParams(max_steps=1000)

vec_env = VmapEnv(BallEnv(), num_envs=512)
obs, states = vec_env.reset(rng, params)              # obs: float32[512, 2]

actions = jnp.zeros(512, dtype=jnp.int32)
obs, states, rewards, dones, infos = vec_env.step(rng, states, actions, params)
# rewards: float32[512]
# dones:   bool[512]
```

### Scan rollout

The canonical training pattern — the entire `N envs × T steps` rollout compiles to a
single fused GPU kernel:

```python
import jax
import jax.numpy as jnp
from envrax import EnvParams
from envrax.wrappers import VmapEnv


@jax.jit
def collect_rollout(rng, params, vec_env, num_steps=128):
    rng, reset_rng = jax.random.split(rng)
    obs, state = vec_env.reset(reset_rng, params)

    def scan_step(carry, _):
        obs, state, rng = carry
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        actions = jax.vmap(lambda r: vec_env.env.action_space.sample(r))(
            jax.random.split(action_rng, vec_env.num_envs)
        )
        obs, state, reward, done, info = vec_env.step(step_rng, state, actions, params)
        return (obs, state, rng), (obs, actions, reward, done)

    _, trajectory = jax.lax.scan(scan_step, (obs, state, rng), None, num_steps)
    return trajectory
```

### `make()` — create with JIT and wrappers

```python
import envrax
import atarax  # registers Atarax envs into envrax on import

# JIT-compiled by default; warm-up step runs at construction time
env, params = envrax.make("atari/breakout-v0")
obs, state = env.reset(jax.random.PRNGKey(0), params)

# Apply wrappers (innermost-first)
from envrax.wrappers import GrayscaleObservation, ResizeObservation
env, params = envrax.make(
    "atari/breakout-v0",
    wrappers=[GrayscaleObservation, ResizeObservation],
    jit_compile=False,
)

# Vectorised environments
vec_env, params = envrax.make_vec("atari/breakout-v0", n_envs=64)
obs, states = vec_env.reset(jax.random.PRNGKey(0), params)  # obs: [64, ...]

# Multiple environments at once
envs = envrax.make_multi(["atari/breakout-v0", "atari/asteroids-v0"])
```

### Registry — low-level lookup

`make_env()` is the bare-metal registry lookup (no JIT, no wrappers):

```python
from envrax import make_env, register, EnvParams

register("BallEnv-v0", BallEnv, EnvParams(max_steps=500))

env, params = make_env("BallEnv-v0")
env, params = make_env("BallEnv-v0", max_steps=1000)  # override default
```

### `JitWrapper` — manual JIT control

```python
from envrax.wrappers import JitWrapper

env = BallEnv()
env = JitWrapper(env)  # compiles reset + step immediately

obs, state = env.reset(jax.random.PRNGKey(0), params)
obs, state, reward, done, info = env.step(jax.random.PRNGKey(1), state, action, params)
```

## Wrappers

Nine generic wrappers compatible with any `JaxEnv`. All expose the same
`reset(rng, params)` / `step(rng, state, action, params)` interface and are fully
compatible with `jit`, `vmap`, and `lax.scan`.

| Wrapper | Input obs | Output obs | Description | Extra state |
| --- | --- | --- | --- | --- |
| `JitWrapper` | any env | same obs | Compiles `reset` + `step` with `jax.jit`; caches kernels to disk | — |
| `GrayscaleObservation` | `uint8[H, W, 3]` | `uint8[H, W]` | NTSC luminance conversion | — |
| `ResizeObservation(h, w)` | `uint8[H, W]` or `uint8[H, W, C]` | `uint8[h, w]` or `uint8[h, w, C]` | Bilinear resize (default 84×84) | — |
| `NormalizeObservation` | `uint8[...]` | `float32[...]` in `[0, 1]` | Divide by 255 | — |
| `FrameStackObservation(n_stack)` | `uint8[H, W]` | `uint8[H, W, n_stack]` | Rolling frame buffer (default 4) | `FrameStackState` |
| `ClipReward` | any reward | `float32 ∈ {−1, 0, +1}` | Sign clipping | — |
| `ExpandDims` | any env | same obs | Adds trailing `1` dim to `reward` and `done` | — |
| `EpisodeDiscount` | any env | same obs | Converts `done` bool to float32 discount (`1.0` / `0.0`) | — |
| `RecordEpisodeStatistics` | any env | same obs | Tracks episode return + length in `info["episode"]` | `EpisodeStatisticsState` |
| `RecordVideo` | any env | same obs | Saves episode frames to MP4 (not JIT-compatible) | — |

Stateless wrappers pass the inner state through unchanged. Stateful wrappers
(`FrameStackObservation`, `RecordEpisodeStatistics`) return a `chex.dataclass` pytree
that wraps the inner state — both are fully compatible with `jit`, `vmap`, and
`lax.scan`.

The `_WrapperFactory` pattern lets parameterised wrappers be used in wrapper lists
without pre-binding an environment:

```python
from envrax.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation

# Each wrapper used as a standalone class
env = GrayscaleObservation(env)
env = ResizeObservation(env, h=84, w=84)
env = FrameStackObservation(env, n_stack=4)
```

## API Reference

### Base classes (`envrax.base`)

| Symbol | Description |
| --- | --- |
| `EnvState` | `chex.dataclass` — `step: int32`, `done: bool`. Extend to add game-specific fields. |
| `EnvParams` | `chex.dataclass` — `max_steps: int = 1000`. Extend to add game-specific config. |
| `JaxEnv` | Abstract base. Implement `reset`, `step`, `observation_space`, `action_space`. |

### Spaces (`envrax.spaces`)

| Symbol | Description |
| --- | --- |
| `Discrete(n)` | `n` integer actions in `[0, n)`. |
| `Box(low, high, shape, dtype)` | Continuous array space. |

### Factory functions (`envrax.make`)

| Symbol | Description |
| --- | --- |
| `make(name, *, params, wrappers, jit_compile, cache_dir)` | Create a single env with optional wrappers and JIT. Returns `(JaxEnv, EnvParams)`. |
| `make_vec(name, n_envs, ...)` | Create a `VmapEnv` of `n_envs` parallel environments. |
| `make_multi(names, ...)` | Create one env per name. Returns `List[(JaxEnv, EnvParams)]`. |
| `make_multi_vec(names, n_envs, ...)` | Create one `VmapEnv` per name. |

### Registry (`envrax.registry`)

| Symbol | Description |
| --- | --- |
| `register(name, cls, default_params)` | Register a `JaxEnv` under a name. Called on package import. |
| `make_env(name, **overrides)` | Bare-metal instantiate by name (no JIT, no wrappers). Returns `(JaxEnv, EnvParams)`. |
| `registered_names()` | Sorted list of all registered environment names. |

## The Envrax Suite

Packages that share this common API:

| Package | PyPI | Description |
| --- | --- | --- |
| **envrax** | `pip install envrax` | Core API, base classes, spaces, wrappers |
| **atarax** | `pip install atarax` | JAX-native Atari 2600 game suite |
<!-- | **proxen** | `pip install proxen` | JAX-native Procgen suite | -->
<!-- | **labrax** | `pip install labrax` | JAX-native DMLab-style 3D navigation | -->

Install only what you need — each suite package pulls in `envrax` automatically.

## Licence

Apache 2.0 — see [LICENSE](LICENSE).
