![Logo](https://raw.githubusercontent.com/Achronus/envrax/main/docs/assets/imgs/main.png)

![Python Version](https://img.shields.io/pypi/pyversions/envrax)
![License](https://img.shields.io/github/license/Achronus/envrax)

Envrax is a lightweight open-source JAX-native Reinforcement Learning (RL) environment API standard for single-agents, equivalent to the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) package. It includes: base classes, spaces, wrappers, and a shared registry for building and utilizing RL environments with ease.

All environment logic follows a *stateless functional design* that builds on top of the [JAX](https://github.com/jax-ml/jax) and [Chex](https://github.com/google-deepmind/chex) packages to benefit from JAX accelerator efficiency.

## Why Envrax?

One of the downsides of RL research is sample efficiency. Often the environment becomes the main bottleneck for model training because they are restricted, and built, around CPU utilisation.

For example, the [Atari](https://ale.farama.org/) suite is CPU constrained and, from our experience, when we increase the number of environments running in parallel, a single training step drastically increases wall-clock time. Gradient computations on a GPU could take ~30 seconds but the sample retrieval takes over 2+ minutes (400% increase) because of the CPU bottleneck and that's with efficiency tricks!

This begged a much deeper question -

> what if we could eliminate the CPU bottleneck by loading the environment onto the same accelerator as the model?

Packages like [Brax](https://github.com/google/brax) and [Gymnax](https://github.com/RobertTLange/gymnax/) have shown the incredible benefits of JAX based environment approaches. However, they are limited to their unique approaches without a unified API standard. Gymnasium has always been a personal favourite of mine because of its API simplicity, but there is no JAX equivalent. Thus, Envrax was born.

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

## API Standard

Envrax enforces a small, strict interface so that every environment, regardless of the suite created, behaves identically under `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

Every environment is built as a stateless Python object and environment states (`envrax.EnvState`) are defined as explicit `chex.dataclass` PyTrees passed to and returned from every call, making the full `reset → step` pipeline compatible with `jax.jit`, `jax.vmap`, and `jax.lax.scan` with zero modification.

At a glance, all environments inherit from the `envrax.JaxEnv` base class and then implement their own `envrax.Spaces`, methods, `envrax.EnvState`, and `envrax.EnvConfig`. By design, `JaxEnv` is generic over four type parameters: the observation space, the action space, the state type, and the config type (`JaxEnv[ObsSpaceT, ActSpaceT, StateT, ConfigT]`) to maximise IDE support.

Here are the core components:

```python
import jax
from envrax import JaxEnv, EnvState, EnvConfig
from envrax.spaces import Box, Discrete

# Core inheritable items
config = EnvConfig()        # static configuration
env = MyEnv(config=config)  # E.g., MyEnv extends JaxEnv with JaxEnv[Box, Discrete, MyEnvState, EnvConfig]

# Required inputs
rng = jax.random.key(42)    # PRNG key (only for reset)

# Core properties
obs_space = env.observation_space
action_space = env.action_space

# Core methods
obs, state = env.reset(rng)  # rng is consumed and stored on state
obs, new_state, reward, done, info = env.step(state, action)
```

This differs slightly from the Gymnasium API standard to maintain JAX compatibility so that we can still trace pure functions without introducing side effects to JIT compilation. Specifically:

- **`config` lives on the env instance**: we set the `config` once at construction so that it never has to be passed to `reset` or `step`.
- **`rng` lives in the state**: our `reset` method consumes a PRNG key and stores its remainder in `state.rng`. The `step` method then splits the `state.rng` internally for any per-step randomness. This means we get to keep the stateless behaviour while threading randomness through the PyTree.

### State and Config as PyTrees

The environment state (`EnvState`) and configuration (`EnvConfig`) are `chex.dataclass` PyTrees. You extend them with game-specific fields such as positions, velocities, timers, while maintaining full compatibility with JAX serialisation, and batched transforms.

In their base forms we have:

```python
@chex.dataclass
class EnvState:
    rng: chex.PRNGKey   # PRNG key threaded through the episode
    step: chex.Array    # current timestep
    done: chex.Array    # episode termination flag

@chex.dataclass
class EnvConfig:
    max_steps: int = 1000  # maximum number of steps per episode
```

The `EnvConfig` acts as static configuration values that are declared once and never changed. While `EnvState` is mutated through the environments lifecycle.

### Generics and Type Safety

Every `JaxEnv` subclass declares its concrete observation, action, state, and config types:

```python
class BallEnv(JaxEnv[Box, Discrete, BallState, BallConfig]): ...
```

This gives you full IDE autocomplete and static type-checking on `env.observation_space`, `env.action_space`, `env.config`, and the `state` returned by `reset`/`step`.

### Spaces

Envrax provides some of the same core space types as Gymnasium (`Discrete`, `Box`, and `MultiDiscrete`) with the same names, semantics, and `sample`/`contains` methods.

Spaces are pure metadata that act as contracts for defining the environment spec. They describe the shapes, bounds, and dtypes for the items used in the RL environment.

| Symbol | Description |
| --- | --- |
| `Discrete(n)` | `n` integer actions in `[0, n)` |
| `Box(low, high, shape, dtype)` | Continuous array space |
| `MultiDiscrete(nvec)` | Vector of independent discrete sub-spaces |

Static fields like `Discrete.n` and `Box.shape` are Python values, so they can be used directly for static decisions in your env logic.

### Wrappers & Composition

Envrax ports several of Gymnasium's most useful wrappers to the JAX-native interface. They follow the same nesting pattern Gymnasium uses where each wrapper takes an inner env and transforms its observations, rewards, or state. Each one follows the standard convention, exposing the same `reset`/`step` signatures as a plain `JaxEnv` but use composition to expand the original environment's functionality without a complete rewrite.

| Wrapper | Kind | Input obs | Output obs | Description |
| --- | --- | --- | --- | --- |
| `JitWrapper` | pass-through | any env | same obs | Compiles `reset` + `step` with `jax.jit`; caches kernels to disk |
| `GrayscaleObservation` | pass-through | `uint8[H, W, 3]` | `uint8[H, W]` | NTSC luminance conversion |
| `ResizeObservation(h, w)` | pass-through | `uint8[H, W]` or `uint8[H, W, C]` | `uint8[h, w]` or `uint8[h, w, C]` | Bilinear resize (default 84×84) |
| `NormalizeObservation` | pass-through | `uint8[...]` | `float32[...]` in `[0, 1]` | Divide by 255 |
| `ClipReward` | pass-through | any reward | `float32 ∈ {−1, 0, +1}` | Sign clipping |
| `ExpandDims` | pass-through | any env | same obs | Adds trailing `1` dim to `reward` and `done` |
| `EpisodeDiscount` | pass-through | any env | same obs | Converts `done` bool to float32 discount (`1.0` / `0.0`) |
| `RecordVideo` | pass-through | any env | same obs | Saves episode frames to MP4 (not JIT-compatible) |
| `FrameStackObservation(n_stack)` | stateful | `uint8[H, W]` | `uint8[H, W, n_stack]` | Rolling frame buffer (default 4); state: `FrameStackState` |
| `RecordEpisodeStatistics` | stateful | any env | same obs | Tracks episode return + length in `info["episode"]`; state: `EpisodeStatisticsState` |

Wrappers come in two flavours:

- **Pass-through**: these preserve the inner state type without changes (e.g. `ClipReward`, `GrayscaleObservation`).
- **Stateful**: these introduce a new outer state type (a `chex.dataclass` extending `EnvState`) that wraps the inner state in an `env_state` field (e.g. `FrameStackObservation`, `RecordEpisodeStatistics`).

Wrappers can be applied either as a list of class instances (no `functools.partial` needed) or composed manually. Envrax handles the rest automatically.

```python
import envrax
from envrax.wrappers import (
    ClipReward,
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)

# Mix of plain classes and pre-configured wrappers — no `partial` needed
env, config = envrax.make(
    "BallEnv-v0",
    wrappers=[
        GrayscaleObservation,
        ResizeObservation(h=84, w=84),
        FrameStackObservation(n_stack=4),
        ClipReward,
    ],
)
```

The same wrappers also work as direct calls if you want to compose them manually:

```python
env = GrayscaleObservation(env)
env = ResizeObservation(env, h=84, w=84)
env = FrameStackObservation(env, n_stack=4)
```

### Registry, Factory & Suite Catalog

Envrax houses a shared registry that lets any installed suite package expose its environments through a single entry point. The registry stores `EnvSpec` objects keyed by canonical names and `make()` methods retrieves them with optional wrappers and JIT compilation.

As Envrax is the base API standard, it ships with zero environments so the registry starts out empty. Environments are contributed by downstream suite packages that call `register()` (or `register_suite()` for bulk registration) at import time. Examples of existing packages will be coming in the future.

The suite catalog is made up of three core components: `EnvSpec`, `EnvSuite`, and `EnvSet`:

| Class | Purpose |
| --- | --- |
| `EnvSpec` | Frozen dataclass holding `(name, env_class, default_config, suite)`. Used for holding the environment of registration. Both `register()` and `register_suite()` build these and store them in the registry. |
| `EnvSuite` | A named, versioned collection of `EnvSpec`s from one suite (e.g. all MuJoCo Playground tasks). They hold a `prefix`, the suite `category`, the suite `version`, its `required_packages`, and a list of specs (`EnvSpec`s). They support slicing, iteration, and package availability checks. |
| `EnvSet` | An ordered collection of `EnvSuite` instances, for users who want to compose multiple suites into one heterogeneous benchmark (e.g. `EnvSet(EnvSuiteA(), EnvSuiteB())`). |

#### Single-env Registration

Use `register()` when you want to add one environment manually:

```python
import envrax
from envrax import EnvConfig

envrax.register("MyEnv-v0", MyEnv, EnvConfig(), suite="my-pkg")
env, config = envrax.make("MyEnv-v0")
```

#### Bulk Registration via a Suite

Use `register_suite()` when shipping a whole benchmark suite. It requires the `EnvSuite.specs` list to be populated to detect new environments for the registry. For example:

```python
from dataclasses import dataclass, field
from typing import List
from envrax import EnvSpec, EnvSuite, register_suite

# Our custom suite of environments
from demo_envs.games.cartpole import CartpoleEnv, CartpoleConfig
from demo_envs.games.ant import AntEnv, AntConfig

@dataclass
class DemoSuite(EnvSuite):
    prefix: str = "demo"
    category: str = "Demo Suite"
    version: str = "v0"
    required_packages: List[str] = field(default_factory=lambda: ["demo_suite"])
    specs: List[EnvSpec] = field(  # Must be populated
        default_factory=lambda: [
            EnvSpec("cartpole", CartpoleEnv, CartpoleConfig()),
            EnvSpec("ant",      AntEnv,      AntConfig()),
        ]
    )

    def get_name(self, name: str, version: str | None = None) -> str:
        return f"{self.prefix}/{name}-{version or self.version}"

# Register every spec in one call — no chance of forgetting one
register_suite(DemoSuite())

# Now usable from anywhere via the standard registry
env, config = envrax.make("demo/cartpole-v0")
```

## Quick Start

### Creating a New Environment

To get started, you first need to create a new environment that inherits from `JaxEnv`. Here's an example:

```python
import chex
import jax
import jax.numpy as jnp

from envrax import JaxEnv, EnvState, EnvConfig
from envrax.spaces import Box, Discrete


@chex.dataclass
class BallState(EnvState):
    ball_x: chex.Array
    ball_y: chex.Array


@chex.dataclass
class BallConfig(EnvConfig):
    friction: float = 0.98
    reward_scale: float = 1.0


class BallEnv(JaxEnv[Box, Discrete, BallState, BallConfig]):
    @property
    def observation_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=4)

    def reset(self, rng: chex.PRNGKey):
        rng, init_rng = jax.random.split(rng)
        rng_x, rng_y = jax.random.split(init_rng)
        state = BallState(
            rng=rng,
            step=jnp.int32(0),
            done=jnp.bool_(False),
            ball_x=jax.random.uniform(rng_x),
            ball_y=jax.random.uniform(rng_y),
        )
        obs = jnp.array([state.ball_x, state.ball_y])
        return obs, state

    def step(self, state: BallState, action: chex.Array):
        rng, _ = jax.random.split(state.rng)

        # Use action to get new obs
        # action: 0=left, 1=right, 2=up, 3=down
        dx = jnp.array([-0.01, 0.01, 0.0, 0.0])[action] * self.config.friction
        dy = jnp.array([0.0, 0.0, -0.01, 0.01])[action] * self.config.friction

        # Get bounds
        low, high = self.observation_space.low, self.observation_space.high

        # Increment obs
        new_x = jnp.clip(state.ball_x + dx, low, high)
        new_y = jnp.clip(state.ball_y + dy, low, high)

        # Update new state
        new_state = state.replace(
            rng=rng,
            step=state.step + 1,
            ball_x=new_x,
            ball_y=new_y,
        )

        # Set new obs
        obs = jnp.array([new_state.ball_x, new_state.ball_y])

        # Compute reward, done, and info
        reward = jnp.float32(1.0) * self.config.reward_scale
        done = new_state.step >= self.config.max_steps
        info = {"current_step": new_state.step}

        return obs, new_state.replace(done=done), reward, done, info
```

This code should work "as is".

### Making Parallel Copies of It

Like Gymnasium's `vector` module, Envrax has it's own `VecEnv` wrapper that can be used to create any `JaxEnv` to run `N` parallel instances via `jax.vmap`. Each environment auto-resets independently when its episode ends.

```python
import jax
import jax.numpy as jnp
from envrax import VecEnv, EnvConfig

env = BallEnv()
vec_env = VecEnv(env, num_envs=512)
obs, states = vec_env.reset(jax.random.key(42))   # obs: float32[512, 2]

actions = jnp.zeros(512, dtype=jnp.int32)
obs, states, rewards, dones, infos = vec_env.step(states, actions)
# rewards: float32[512]
# dones:   bool[512]
```

This code should work "as is" with the custom `BallEnv`.

### Managing Multiple Environments

Envrax also comes out-of-the-box with multi environment handling. This is useful for meta-learning, multi-task training, or any scenario where you need `M` different environments running simultaneously, use `MultiEnv` or `MultiVecEnv`:

```python
import jax
import envrax

# Create M heterogeneous environments (different classes, configs, shapes)
# pre_warm=False by default — compilation is deferred
multi = envrax.make_multi(["BallEnv-v0", "CartPole-v0", "BallEnv-v0"])

# Compile all JIT-wrapped envs in one setup phase (with progress bar)
multi.compile()

# Reset all M envs with a single PRNG key (split internally)
obs_list, states = multi.reset(jax.random.key(0))

# Step all M envs
actions = [jnp.int32(0) for _ in range(multi.num_envs)]
obs_list, states, rewards, dones, infos = multi.step(states, actions)

# Reset a single env (e.g., when its lifetime budget expires)
obs_list[0], states[0] = multi.reset_at(0, jax.random.key(1))
```

`MultiVecEnv` follows the same pattern but wraps `VecEnv` instances:

```python
multi_vec = envrax.make_multi_vec(["BallEnv-v0", "CartPole-v0"], n_envs=64)
multi_vec.compile()

obs_list, states = multi_vec.reset(jax.random.key(0))
# obs_list[0].shape == (64, ...)  — each element is already batched
```

Both classes return lists of values (not stacked arrays) since heterogeneous envs may have different observation shapes. Use `multi.class_groups` to identify which indices share a class for downstream batching.

### `make()` — create with JIT and wrappers

```python
import jax
import envrax
from envrax import EnvConfig

# Register our custom env (suite packages do this on import)
envrax.register("BallEnv-v0", BallEnv, EnvConfig(max_steps=500))

# JIT-compiled by default; warm-up step runs at construction time
env, config = envrax.make("BallEnv-v0")
obs, state = env.reset(jax.random.key(0))

# Apply wrappers (innermost-first)
from envrax.wrappers import NormalizeObservation, ClipReward
env, config = envrax.make(
    "BallEnv-v0",
    wrappers=[NormalizeObservation, ClipReward],
    jit_compile=False,
)

# Vectorised environments
vec_env, config = envrax.make_vec("BallEnv-v0", n_envs=64)
obs, states = vec_env.reset(jax.random.key(0))         # obs: [64, ...]

# Multiple unique environments at once (pre_warm=False by default)
multi = envrax.make_multi(["BallEnv-v0", "ExtraEnv-v0"])
multi.compile()  # separate setup phase
```

### Training Loop

Same simple training loop as [Gymnasium](https://gymnasium.farama.org/) but JAXified!

```python
import envrax
import jax

# Init the environment
env = envrax.make("BallEnv-v0")

# Set it's initial state
key = jax.random.key(42)
obs, state = env.reset(key)

# Iterate through 1000 timesteps
for _ in range(1000):
    action = env.action_space.sample(key)
    obs, state, reward, done, info = env.step(state, action)

    # If episode has ended, reset to start a new one
    if done:
        new_key, key = jax.random.split(key)
        obs, info = env.reset(new_key)
```

### `JitWrapper` — manual JIT control

```python
import jax
from envrax.wrappers import JitWrapper

# Compile immediately (default)
env = JitWrapper(BallEnv())
obs, state = env.reset(jax.random.key(0))

# Defer compilation to a separate setup phase
env = JitWrapper(BallEnv(), pre_warm=False)
env.compile()  # trigger XLA compilation explicitly
obs, state = env.reset(jax.random.key(0))
```

`VecEnv` also exposes a `compile()` method for the same purpose:

```python
vec_env = VecEnv(BallEnv(), num_envs=64)
vec_env.compile()  # warm up the vmapped reset + step
```

## API Reference

### Core Classes (`envrax.env`)

| Symbol | Description |
| --- | --- |
| `EnvState` | `chex.dataclass` — `rng: PRNGKey`, `step: int32`, `done: bool`. Extend to add game-specific fields. |
| `EnvConfig` | `chex.dataclass` — `max_steps: int = 1000`. Extend to add game-specific config. |
| `JaxEnv[ObsSpaceT, ActSpaceT, StateT]` | Generic abstract base. Implement `reset`, `step`, `observation_space`, `action_space`. |

### Factory Functions (`envrax.make`)

| Symbol | Description |
| --- | --- |
| `make(name, *, config, wrappers, jit_compile, pre_warm, cache_dir)` | Create a single env with optional wrappers and JIT. Returns `(JaxEnv, EnvConfig)`. |
| `make_vec(name, n_envs, ...)` | Create a `VecEnv` of `n_envs` parallel environments. Returns `(VecEnv, EnvConfig)`. |
| `make_multi(names, ...)` | Create a `MultiEnv` managing `M` heterogeneous environments. `pre_warm` defaults to `False`. |
| `make_multi_vec(names, n_envs, ...)` | Create a `MultiVecEnv` managing `M` heterogeneous vectorised environments. `pre_warm` defaults to `False`. |

### Multi-Env Managers (`envrax.multi_env`, `envra.multi_vec_env`)

| Symbol | Description |
| --- | --- |
| `MultiEnv(envs)` | Manages M heterogeneous `JaxEnv` instances. `reset(rng)`, `step(states, actions)`, `reset_at(idx, rng)`, `step_at(idx, state, action)`. Returns lists. |
| `MultiVecEnv(vec_envs)` | Manages M heterogeneous `VecEnv` instances. Same API as `MultiEnv`, but each element is already batched. |
| `.compile(progress=True)` | Trigger XLA compilation for all inner envs/VecEnvs with an optional `tqdm` progress bar. |
| `.class_groups` | `Dict[str, List[int]]` — env class name to indices, for downstream same-class batching. |

### Registry & Catalog (`envrax.registry`, `envrax.envs`)

| Symbol | Description |
| --- | --- |
| `EnvSpec(name, env_class, default_config, suite="")` | Frozen dataclass — the unit of registration. Stored in the registry under its canonical name. |
| `EnvSuite` | Base class for declaring a suite of environments. Subclasses pin `prefix`, `category`, `version`, `required_packages`, and a `specs: List[EnvSpec]`. Override `get_name()` to produce canonical IDs. |
| `EnvSet(*suites)` | Collection of `EnvSuite` instances. Supports `+`, iteration, and `verify_packages()`. |
| `register(name, cls, default_config, *, suite="")` | Register a single `JaxEnv` under a name. Builds an `EnvSpec` internally. |
| `register_suite(suite, *, version=None)` | Register every spec in an `EnvSuite` under its canonical IDs. |
| `get_spec(name)` | Return the full registered `EnvSpec` for an environment. |
| `registered_names()` | Sorted list of all registered environment names. |
