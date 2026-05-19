# Vectorising with `VecEnv`

???+ api "API Docs"

    [`envrax.vec_env.VecEnv`](../../api/env/vec.md#envrax.vec_env.VecEnv)

Excellent work building your first environment! If you wanted to, you could stop there and start using Envrax in your own projects right **now** for your own RL experiments, but a single environment is quite... inefficient.

Think about it - a single `JaxEnv` runs one environment for one episode at a time. If you wanted to run over 1 million timesteps to train your policy, that's incredibly sample inefficient and could take weeks to finish training.

What we really need, is a way to make multiple copies of it with randomization automatically built in. Well, that's where `VecEnv` comes in! :wink:

We can wrap the environment in `VecEnv` and it will operate on a *batch* of `N` independent environments simultaneously via `jax.vmap`.

No process pools, no pickling, no cross-device transfers, just pure JAX-native vectorisation for maximum performance! :rocket:

Here's an example:

```python hl_lines="8"
import jax
import jax.numpy as jnp

from envrax import VecEnv, EnvConfig
from my_project import BallEnv

env = BallEnv()
vec_env = VecEnv(env, num_envs=512)

obs, states = vec_env.reset(jax.random.key(42))
# obs:    float32[512, 2]
# states: BallState with all fields shaped [512, ...]

actions = jnp.zeros(512, dtype=jnp.int32)
obs, states, rewards, dones, infos = vec_env.step(states, actions)
# rewards: float32[512]
# dones:   bool[512]
```

We create an instance of our environment, pass it into `VecEnv` and provide a number of `num_envs` to create. It then does the rest without any tweaks to the API! It's that simple! :smile:

## How It Works

`VecEnv` is ~30 lines of glue around `jax.vmap`, and the core implementation of the [`BatchedEnv`](../../api/env/batched.md) base class. Here's a quick rundown:

1. **`reset(rng)`** — splits `rng` into `num_envs` sub-keys and vmaps the inner `env.reset` over them.
2. **`step(state, actions)`** — vmaps `env.step` over the batched state and actions.
3. **Environments auto-reset** — after each `step`, any env with `done=True` is automatically reset using the next rng from its own state. The selection between "step output" and "fresh reset" is done with a `jnp.where` mask inside the vmapped body, so episode boundaries don't require Python-level control flow.

The auto-reset behaviour is what makes `VecEnv` "training-ready": you never have to branch on `done` yourself when collecting rollouts; it does it all for you! :smile:

??? info "What's `BatchedEnv`?"
    `BatchedEnv` is the abstract base class that says "I step `N` independent agent results in one call." `VecEnv` is one strategy that satisfies it (vmap over a single `JaxEnv`); downstream packages can ship others (e.g. a composite MJX scene). [`MultiVecEnv`](multi-env.md#multivecenv) accepts any `BatchedEnv`, so the underlying batching strategy stays pluggable.

## Available Attributes

Just like [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/), Envrax's `VecEnv` provides a small set of attributes and properties that may come in handy during training:

| Item | Description |
| --- | --- |
| `vec_env.env` | The wrapped inner `JaxEnv` |
| `vec_env.num_envs` | The number of parallel environments |
| `vec_env.n_slots` | Alias for `num_envs`, satisfies the `BatchedEnv` contract |
| `vec_env.name` | Inner env's class name — used as the default key by `MultiVecEnv` |
| `vec_env.config` | The inner environment's config for quick and easy access |
| `vec_env.single_observation_space` | The per-env observation space |
| `vec_env.single_action_space` | The per-env action space |
| `vec_env.observation_space` | The batched observation space with a leading `num_envs` dim (`B`) |
| `vec_env.action_space` | The batched action space with a leading `num_envs` dim (`B`) |

## JIT Compiling

By default, JAX compiles a function lazily on its first real call. For a `VecEnv`, the first `step` kicks off XLA compilation and can take anywhere from a couple of seconds up to a minute, depending on env complexity.

This cost can be pretty annoying during a training run, so we've added a `compile()` method to `VecEnv`. With this, you can create your own setup stages in advance, and cache the XLA-compiled kernels (default: `.jax_cache` in the project root) too! :wink:

Here's how to use it:

```python
vec_env = VecEnv(BallEnv(), num_envs=512)
vec_env.compile()    # runs a dummy reset + step to trigger XLA
# ... later, in training:
obs, states = vec_env.reset(jax.random.key(0))
```

With caching in place, subsequent Python processes will reuse the same compiled kernels to drastically reduce future compiling time. This is useful for test runs, when you need to stop and start a training run, or when your program unexpectedly crashes. Those precious seconds make all the difference!

## Using Wrappers

As you'll see in a future tutorial ([Available Wrappers](wrappers.md)), Envrax comes with a host of environment wrappers out-of-the-box.

To use them with `VecEnv`, you need to apply them to your `JaxEnv` first, then pass the wrapped environment to `VecEnv`:

```python
from envrax import VecEnv
from envrax.wrappers import GrayscaleObservation, FrameStackObservation

env = BallEnv()
env = GrayscaleObservation(env)
env = FrameStackObservation(env, n_stack=4)

vec_env = VecEnv(env, num_envs=512)   # wrappers live inside the vmap
```

This order matters for two reasons:

1. **Wrappers transform per-env data** - the `GrayscaleObservation` wrapper expects `uint8[H, W, 3]`, not `uint8[N, H, W, 3]`. Putting it outside `VecEnv` would feed it batched arrays it can't handle.
2. **`VecEnv` isn't a `JaxEnv`** - wrappers expect a `JaxEnv` instance as their inner env and `VecEnv` isn't a `JaxEnv`, it's a basic class wrapper around it.

The [`make_vec()`](make.md) factory method applies wrappers in this order automatically. We'll cover the full set of factory methods later in the [Make Methods](make.md) tutorial! :wink:

!!! warning "`RecordVideo` is the Exception"
    `RecordVideo` writes MP4 files Python-side and is **not JIT/vmap-compatible**. Use it on a single environment, or render manually via `vec_env.render_slot(states, slot_idx=0)` and feed an external recorder.

## Rendering

`VecEnv` exposes `render_slot()` and `slot_state()` for pulling a single environment out of the batch. `render_slot` extracts one slot and delegates to the inner environment's `render`:

```python
frame = vec_env.render_slot(states, slot_idx=0)    # np.ndarray uint8 (H, W, 3)
single = vec_env.slot_state(states, slot_idx=0)    # unbatched state pytree
```

This is useful for logging an episode during training without unpacking the batched state yourself, and for any downstream tooling that wants a single agent's state at a time. We'll discuss [Rendering](rendering.md) more in a future tutorial.

## Common Pitfalls

Like `EnvState`, there are a few common "gotchas" to be mindful of:

- **Mismatched action shape** — `actions` must have shape `(num_envs, ...)` with the same dtype as the action space. For a `Discrete` action, that's `jnp.int32[num_envs]`.
- **`reset` with a single key** — `VecEnv.reset` takes one master key and splits it internally automatically. Don't pre-split your keys!
- **Trying to use Python-side side-effects inside `step`** — `VecEnv` vmaps over the batch, so `print()`, file writes, etc. trace and explode.
- **Forgetting `compile()` in benchmarks** — the first call will always look slow because XLA is compiling. Call `compile()` before timing anything.

## Recap

To recap:

- `VecEnv(env, num_envs)` uses `jax.vmap` on your environment for batched rollouts
- `VecEnv` inherits from `BatchedEnv` — it's the canonical vmap strategy for that contract
- Batched fields all gain a leading `num_envs` dimension
- Auto-reset on `done=True` is handled inside the vmapped body via a `jnp.where` mask — no Python control flow needed
- A small set of attributes (`env`, `num_envs`, `n_slots`, `name`, `config`, plus single and batched space properties) gives quick access to the wrapped environment's metadata
- Call `vec_env.compile()` to trigger XLA compilation as a separate setup phase, with cached kernels reused across runs
- Apply wrappers to your `JaxEnv` *first*, then pass the wrapped environment to `VecEnv`
- `vec_env.render_slot(states, slot_idx=0)` extracts one environment from the batch for visual inspection; `vec_env.slot_state(states, slot_idx=0)` gives you the raw unbatched state pytree

Next, we'll look at using some new classes to make `M` *heterogeneous* environments with ease. See you there! :wave:

## Next Steps

<div class="grid cards" markdown>

-   :material-group:{ .lg .middle } __Multiple Environments__

    ---

    Learn how to manage `M` *heterogeneous* environments with `MultiEnv` / `MultiVecEnv`.

    [:octicons-arrow-right-24: Continue to Tutorial 6](multi-env.md)

</div>
