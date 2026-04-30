# Environment State

Welcome to your first Envrax tutorial! :wave:

Before we build anything, we first need to understand two foundational concepts: the environment's [state](#what-is-a-state) and environment [spaces](spaces.md). In this tutorial, we'll focus on its [state](#what-is-a-state).

## What is a State?

In its simplest form, an environment state is a single snapshot of the current internal representation of the environment that provides a full description of the world.

This is distinct from two related concepts often used in the RL setting: `observations` and `dynamics`. Here's how:

1. Observations - are a _subset_/_transformation_ of the state, limiting what the _agent_ gets to see each step.
2. Dynamics - are the _rules_ of the environment that compute the next state from the current one.

For a ball moving in 2D, the environment state might contain:

- The ball's current position
- Its velocity
- How many steps have passed
- Whether the episode has ended

If we know the state of the environment, we can then compute the next state given an action, following the Markov property that RL algorithms use.

## The Base State

???+ api "API Docs"

    [`envrax.env.EnvState`](../../api/env/base.md#envrax.env.EnvState)

By design, Envrax represents state as a [`@chex.dataclass` [:material-arrow-right-bottom:]](https://chex.readthedocs.io/en/latest/api.html#dataclasses) — an _immutable_ Python object that JAX treats as a ["PyTree" [:material-arrow-right-bottom:]](https://docs.jax.dev/en/latest/pytrees.html). This allows us to work with the JAX package without any issues and enables `jax.vmap` with thousands of environments at once.

??? note "But really, why `@chex.dataclass`?"
    As mentioned, it registers your class as a JAX PyTree, which gives you four things for free:

    1. **Automatic traversal** by `jax.jit`, `jax.vmap`, `jax.lax.scan`, etc.
    2. **`.replace(...)`** for immutable updates
    3. **Batching** — `VecEnv` can stack `N` states into a single PyTree with a leading batch dimension
    4. **Testing helpers** — works out of the box with `chex`'s assertion utilities (`chex.assert_tree_all_close`, `chex.assert_shape`, etc.) for verifying state transitions in unit tests

    Plain `@dataclass`es won't work because they _are not_ PyTrees so JAX can't trace them!

Every Envrax state _must_ inherit from `envrax.EnvState`. By default, it provides three mandatory fields essential to all environments:

```python
import chex
import jax.numpy as jnp

@chex.dataclass
class EnvState:
    rng: chex.PRNGKey  # (1)
    step: chex.Array   # (2)
    done: chex.Array   # (3)
```

1. `jax.random.key()` threaded through the episode
2. The current environment timestep (`int32`)
3. Environment termination flag (`bool`)

Using class inheritance, you can extend it with whatever your environment needs and keep those fields for free! :muscle:

Sticking to our 2D ball example, we could add its current `x` and `y` position like so:

```python
import chex
import jax.numpy as jnp

from envrax import EnvState

@chex.dataclass
class BallState(EnvState):
    ball_x: chex.Array
    ball_y: chex.Array
```

Notice how we don't use the Python `float` type here. There's a reason for that and we'll explain that shortly.

But now, whenever we use `BallState`, we have access to all five fields: `rng`, `step`, `done`, `ball_x` and `ball_y`.

We'll use this `BallState` throughout the next couple of tutorials, so make sure you get familiar with it!

## Field Rules/Types

??? note "Chex Arrays"
    `chex.Array` is a type alias for JAX and NumPy arrays making it a convenient annotation for "any array-like field". It doesn't wrap or modify values at runtime; it just makes type hints more readable.

    For consistency, and convenience, we use them throughout the tutorials anywhere a field holds an array.

Fields on an `EnvState` subclass must be **JAX-compatible** and **traceable**. This means we can have either:

- [x] JAX arrays (`jnp.float32`, `jnp.int32`, `jnp.bool_`, `jax.ndarray`, `chex.Array`, etc.)
- [x] Nested `@chex.dataclass` instances
- [ ] Python `list`, `dict`, `tuple`
- [ ] Python objects, strings, `None`
- [ ] Python `int`, `float`, `bool`

**Traceable** values are really important for the flow of JAX JIT-compiled functions. They act as runtime data, allowing them to be changed during each function call without triggering a re-compile.

JIT-compiling can take a lot of time depending on the size of the _computation graph_ so we really only want 1 "setup" compile at the start of using an environment to help us drastically reduce wall-clock time.

We cannot use Python types like `int`, `float` and `bool` because they are **static** values. Every time they change, they need to be re-traced and re-compiled. These are great for [`EnvConfig`](configuration.md) instead - more on them in a later tutorial!

If you need a fixed-size collection in your `EnvState`, use a JAX array:

```python
@chex.dataclass
class SnakeState(EnvState):
    body_positions: chex.Array   # shape: (max_length, 2)
    body_length: jnp.int32       # how many rows of body_positions are valid
```

??? warning "Array Shapes"
    Array sizes **must** be a fixed shape. If your logical length varies, pad to a max and track a valid-length scalar.

    You want to avoid re-compiling whenever possible!

## Updating State

Since PyTrees are immutable, we have to use the built-in `chex.dataclass` method `.replace(...)` whenever we want to make state adjustments.

This returns a new state with the requested fields changed and copies the other fields over automatically:

```python
state = BallState(
    rng=jax.random.key(0),
    step=jnp.int32(0),
    done=jnp.bool_(False),
    ball_x=jnp.float32(0.1),
    ball_y=jnp.float32(0.1),
)

# Updating two fields
next_state = state.replace(
    ball_x=state.ball_x + 0.05,
    step=state.step + 1,
)
```

Keeping to the JAX theme, our state transitions stay "pure" for JIT compatibility, without making you rebuild every field by hand.

## Threading the PRNG Key

RL environments need randomness for: random starting positions, noisy transitions, and stochastic rewards.

JAX handles randomness through [PRNG keys [:material-arrow-right-bottom:]](https://docs.jax.dev/en/latest/jax.random.html#prng-keys). These are explicit values that you _split_ before consuming, rather than needing a hidden global state stored within your environment. We make these keys with the [`jax.random.key()` [:material-arrow-right-bottom:]](https://docs.jax.dev/en/latest/_autosummary/jax.random.key.html#jax.random.key) method.

Envrax threads a key through the episode by storing it on the state. The pattern is always the same:

```python
rng, sub = jax.random.split(state.rng)
noise = jax.random.normal(sub, shape=(2,)) * 0.01
new_state = state.replace(rng=rng, ball_x=state.ball_x + noise[0])
```

One half of the split (`rng`) goes back on the state for the next call, while the other half (`sub`) is consumed _now_ for this step's randomness.

??? warning "Never Reuse a Key!"
    Reusing the same `rng` twice gives you the _same_ sample twice. This is a common source of silent determinism bugs.

    :exclamation: **Always** split your keys before consuming them! :exclamation:

We'll explore this in more detail when we put this into a real `step` method in the ["Your First Environment"](first-env.md) tutorial. For now, just remember the split-then-consume pattern.

## Nested States

Sometimes environment logic might need to _remember_ something between steps (e.g., a rolling buffer of frames or a running reward total). Rather than mutating the inner state, we can _wrap_ it in a larger state with its own extra fields.

The inner state stays untouched and we can still read its information whenever we need it.

A common pattern for this is stacked `Wrappers`. If you apply a wrapper on top of an environment, the wrapper needs to be able to read the environment's base fields (`rng`, `step`, `done`).

The pattern is similar to what we've discussed previously that uses the `@chex.dataclass` decorator, but we now have an `env_state` field that provides us access to an "inner" `EnvState`:

```python
@chex.dataclass
class FrameStackState(EnvState):
    env_state: EnvState       # forwarded inner state
    obs_stack: chex.Array     # wrapper-specific data
```

That's it! Everything the inner environment provided is preserved, plus whatever its wrapper needs to remember. This is a more advanced topic so we'll build on this in a later tutorial.

For those curious, you can check it out in the [Creating a Custom Wrapper](../advanced/custom-wrapper.md) tutorial.

## Common Pitfalls

When building your custom `EnvState`s, there are a few common "gotchas" to be mindful of:

- **`AttributeError: 'BallState' object has no attribute 'replace'`** — you forgot to add the `@chex.dataclass`.
- **`TypeError: Argument ... is not a valid JAX type`** — a field is a Python object or `None` value. Convert it to a JAX array.
- **Silent determinism bugs** — `reset` was called twice with the _same_ key and produced the "same episode" that you expected to be different. Make sure you are using [`jax.random.split` [:material-arrow-right-bottom:]](https://docs.jax.dev/en/latest/_autosummary/jax.random.split.html) per environment.
- **Shape mismatches under `vmap`** — a field has a Python `int` instead of a JAX scalar. Wrap with `jnp.int32(...)`.

## Recap

And that covers the basics of `EnvState`! Great job getting this far! :partying_face:

To recap:

- `EnvState` is a full snapshot of the environment at one timestep
- `EnvState` fields must be both **JAX-compatible** and **traceable** — use `jnp.*` types, not Python `int`/`float`/`bool`
- In Envrax, `EnvState` is **immutable** because JAX needs pure functions, so we `.replace(...)` rather than mutate
- We extend `EnvState` with `@chex.dataclass` and JAX-compatible fields
- We thread the PRNG key through the episode by splitting it each step
- We can nest one `EnvState` inside another when wrappers need extra state of their own

## Next Steps

Next, we'll look at the second foundational concept of Envrax - `spaces`!

<div class="grid cards" markdown>

-   :material-shape-outline:{ .lg .middle } __Spaces__

    ---

    Learn how to describe observations and actions with `Spaces`.

    [:octicons-arrow-right-24: Continue to Tutorial 2](spaces.md)

</div>
