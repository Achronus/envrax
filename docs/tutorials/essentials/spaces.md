# Spaces

Now that you've seen how [state](state.md) describes the _internal_ world of your environment, we'll move on to our next concept and talk about the _interface_ an agent sees. That's what **spaces** are for.

## What is a Space?

A space is a **contract** that describes the shape, bounds, and dtype of the observations and actions flowing in and out of your environment.

You can think of it like a type signature for RL environment data that tells the agent what it can _see_ (`observations`) and how it can _act_ (`actions`).

Every environment **must** have the following:

- **Observation space** - what the agent sees (`env.observation_space`)
- **Action space** - how the agent interacts with the environment (`env.action_space`)

## Why Use Them?

For three reasons:

1. **Agents can easily shape their policy networks** - a policy needs to know the action space to build its output head, and the observation space to build its input layer.
2. **We can easily expand environments using wrappers** - certain Envrax built-in wrappers like the `GrayscaleObservation` wrappers checks that its observation input is `uint8[H, W, 3]`. Without spaces, we'd have to add extra logic within our training loop.
3. **You can catch bugs early** - if your env claims `Box(0, 1, (4,))` but actually returns shape `(3,)`, tests can verify the contract in seconds.

## Built-In Spaces

???+ api "API Docs"

    [`envrax.spaces.Space`](../../api/spaces.md#envrax.spaces.Space)

Envrax ships with three space types: `Discrete`, `Box` and `MultiDiscrete`.

All of them implement three methods from the `Space` contract:

- **`sample(rng)`** — draw a random element
- **`contains(x)`** — check if `x` is a valid element of the space
- **`batch(n)`** — return a batched version of the space with a leading dimension `n`.

### Discrete

???+ api "API Docs"

    [`envrax.spaces.Discrete`](../../api/spaces.md#envrax.spaces.Discrete)

`Discrete` spaces are one of the simplest available and are commonly used for deterministic problem sets.

Here's some example use cases:

- Action space - agent moves with 4 movements: `[up, right, down, left]`
- Observation space - environment is a 4x4 grid world of indices `[x, y]`

We can make one like so:

```python
from envrax.spaces import Discrete

space = Discrete(n=4)  # actions: 0, 1, 2, 3

# Properties
space.dtype            # jnp.int32 (default) - space data type
space.n                # 4 - number of available actions

# Methods
action = space.sample(jax.random.key(0))   # E.g., int32(2)
space.contains(action)                     # True
space.batch(8)                             # MultiDiscrete(nvec=(4,)*8, dtype=jnp.int32)
```

Because `n` is a static Python `int`, you can use it directly in shape declarations or `jnp.arange(space.n)` without issues.

### Box

???+ api "API Docs"

    [`envrax.spaces.Box`](../../api/spaces.md#envrax.spaces.Box)

`Box` spaces are another common type that are often used for continuous-valued observations or actions with per-dimension bounds.

When comparing `Box` to `Discrete`, `Box` focuses on _continuous ranges with bounds_ while `Discrete` focuses on _counting_ based approaches.

```python
from envrax.spaces import Box
import jax.numpy as jnp

space = Box(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)

# Properties
space.dtype   # jnp.float32 - space data type
space.low     # 0.0 - Scalar lower bound applied to all elements
space.high    # 1.0 - Scalar upper bound applied to all elements
space.shape   # (2,) - tuple describing a single element's shape

# Methods
action = space.sample(jax.random.key(0))   # E.g., jnp.float32((2,))
space.contains(action)                     # True
space.batch(8)                             # Box(low=0.0, high=1.0, shape=(8, 2), dtype=jnp.float32)
```

Integer dtypes are also supported:

```python
# Image observations
Box(low=0, high=255, shape=(84, 84, 3), dtype=jnp.uint8)
```

### MultiDiscrete

???+ api "API Docs"

    [`envrax.spaces.MultiDiscrete`](../../api/spaces.md#envrax.spaces.MultiDiscrete)

`MultiDiscrete` is less common than the others and is used when an action is a _vector_ of independent discrete choices, E.g., a game pad with a directional stick (4 options) and two buttons (with 2 options each):

```python
from envrax.spaces import MultiDiscrete

space = MultiDiscrete(nvec=(4, 2, 2))

# Properties
space.dtype   # jnp.int32 (default) - space data type
space.nvec    # (4, 2, 2) - tuple of action counts
space.shape   # (3,) - tuple describing the spaces shape

# Methods
action = space.sample(jax.random.key(0))   # E.g., int32[3] — one pick per sub-space
space.contains(action)                     # True
space.batch(2)                             # MultiDiscrete(nvec=(4, 2, 2, 4, 2, 2), dtype=jnp.int32)
```

Each element `i` of the sampled action satisfies `0 <= action[i] < nvec[i]`.

## Picking the Right Space

A quick decision tree:

| Your data is... | Space |
| --- | --- |
| One categorical choice (`"up"`, `"down"`, ...) | `Discrete` |
| A continuous array (positions, velocities, pixels) | `Box` |
| A _vector_ of independent categorical choices | `MultiDiscrete` |

If none fit, you're probably modelling something more exotic (e.g. a `Tuple`, or `Dict`) that Envrax doesn't currently support. In this case, there are two options:

1. Encode it as a flat `Box` or `MultiDiscrete` and decode it yourself inside your environment.
2. Build your own by subclassing `Space` and implementing `sample`/`contains`/`batch`. You can learn more about this in the advanced tutorial - [Creating a Custom Space](../advanced/custom-spaces.md).

## Recap

And that's that! Nice job :smile:! Let's quickly recap:

- Spaces are **contracts** that describe the shape and bounds of observations and actions
- Use `Discrete(n)` for a single categorical choice
- Use `Box(low, high, shape, dtype)` for continuous arrays or images
- Use `MultiDiscrete(nvec)` for a vector of independent categorical choices

All three `Space` methods — `sample(rng)`, `contains(x)`, and `batch(n)` — are available on every space, ready for use in testing, wrappers, and `VecEnv`.

## Next Steps

Two foundational pieces down. Next up: how environments hold their static, per-env settings via `EnvConfig`!

<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle } __Environment Configuration__

    ---

    Learn how to extend `EnvConfig` with your own static fields and how it differs from `EnvState`.

    [:octicons-arrow-right-24: Continue to Tutorial 3](configuration.md)

</div>
