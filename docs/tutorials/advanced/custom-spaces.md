# Creating a Custom Space

In our [Spaces](../essentials/spaces.md) tutorial we discussed how Envrax ships with three space types: `Discrete`, `Box`, and `MultiDiscrete`. These are the simplest variants and cover the most common RL environment use cases, but sometimes you need something a lot more unique.

Maybe you want a one-hot encoded categorical space, a bitstring, a truncated normal distribution, or a weighted distribution. None of those are obvious fits for the current built-ins spaces. So, you'll need to build your own!

In this tutorial, we'll walk through how to do exactly that. Let's get into it! :muscle:

## `Space` Requirements

???+ api "API Docs"

    [`envrax.spaces.Space`](../../api/spaces.md#envrax.spaces.Space)

Every space **must** inherit from the `envrax.spaces.Space` base class and implement three methods:

| Method | Purpose | Returns |
| --- | --- | --- |
| `sample(rng)` | Samples a random action from the space | `chex.Array` |
| `contains(x)` | Checks if `x` is a valid item in the space | `bool` |
| `batch(n)` | Returns a space with a leading batch dimension `n` | `Space` |

We also recommend making the custom space class a frozen dataclass using `@dataclasses.dataclass(frozen=True)` to make it immutable metadata. Its a useful practice to help avoid accidental changes to something that should be a static entity. Envrax also does this with its own built-in spaces.

Okay, so that's the basics of `Space` requirements. Let's now build a custom one to get a better feel for it.

## Working Example: `OneHot(n)`

??? example "Full Code"

    ```python
    from dataclasses import dataclass
    from typing import Self, Tuple, Type

    import chex
    import jax
    import jax.numpy as jnp

    from envrax.spaces import Space


    @dataclass(frozen=True)
    class OneHot(Space):
        """
        One-hot encoded categorical over `n` classes.

        Parameters
        ----------
        n : int
            Number of categories.
        batch_shape : Tuple[int, ...] (optional)
            Leading shape for batched sampling. Defaults to `()` (unbatched).
        dtype : Type (optional)
            Element dtype. Defaults to `jnp.float32`.
        probs : Tuple[float, ...] | None (optional)
            Per-category sampling probabilities. Must have length `n` and sum
            to 1. When `None` (the default), `sample()` draws uniformly.
        """

        n: int
        batch_shape: Tuple[int, ...] = ()
        dtype: Type = jnp.float32
        probs: Tuple[float, ...] | None = None

        def sample(self, rng: chex.Array) -> chex.Array:
            """
            Sample a random one-hot vector of shape `(*batch_shape, n)`.

            Parameters
            ----------
            rng : chex.Array
                JAX PRNG key.

            Returns
            -------
            action : chex.Array
                One-hot encoded action of shape `(*batch_shape, n)` and the
                space's `dtype`.
            """
            if self.probs is not None:
                idx = jax.random.choice(
                    rng,
                    self.n,
                    shape=self.batch_shape,
                    p=jnp.array(self.probs),
                )
            else:
                idx = jax.random.randint(
                    rng,
                    shape=self.batch_shape,
                    minval=0,
                    maxval=self.n,
                )
            return jax.nn.one_hot(idx, self.n, dtype=self.dtype)

        def contains(self, x: chex.Array) -> bool:
            """
            Check that `x` is a valid one-hot tensor of the expected shape.

            Parameters
            ----------
            x : chex.Array
                Action to validate. Expected to be a one-hot vector of
                shape `(*batch_shape, n)`.

            Returns
            -------
            valid : bool
                `True` if `x` matches the expected shape, contains only
                `0`s and `1`s, and has exactly one `1` along the last axis.
            """
            expected_shape = (*self.batch_shape, self.n)
            if x.shape != expected_shape:
                return False

            is_binary = jnp.all((x == 0) | (x == 1))
            has_one_hot = jnp.all(jnp.sum(x, axis=-1) == 1)
            return bool(is_binary & has_one_hot)

        def batch(self, n: int) -> Self:
            """
            Return a `OneHot` with a leading `n` dimension prepended to
            `batch_shape`.

            Parameters
            ----------
            n : int
                Batch size to prepend to the existing `batch_shape`.

            Returns
            -------
            batched : OneHot
                A new `OneHot` with `batch_shape=(n, *self.batch_shape)`.
                All other fields (`dtype`, `probs`) are forwarded unchanged.
            """
            return OneHot(
                n=self.n,
                batch_shape=(n, *self.batch_shape),
                dtype=self.dtype,
                probs=self.probs,
            )
    ```

For this tutorial, we'll build a **one-hot encoded categorical** action space.

For those unfamiliar with the concept, we take a standard set of categorical options and turn each one into a vector of `0`s with a single `1` at its respective index.

Let's say we have 3 actions: `up`, `down`, and `stay`. Each one represents a different category. As a vector, they would look like this:

| up | down | stay |
|----|------|------|
| 1  |  0   |  0   |
| 0  |  1   |  0   |
| 0  |  0   |  1   |

That's it! That's what our space would hold. For now, we'll call the number of our categories `n` and think about how we can implement this later.

Next, we need to consider how we want to support `VecEnv` compatibility. For the `batch(n)` method we need a way to define and store the batch dimension.

With Envrax's `Discrete` space, we turned it into a `MultiDiscrete` space. That's one way of doing it (turning the space into another one entirely), but it feels a bit overkill here. Instead, we can just track the leading shape dimension for our batch as a `Tuple[int, ...]`. We'll call this our `batch_shape`. Again, we'll get into this in a bit more detail shortly.

That's our two parameters sorted. The next question is: what can this space actually be useful for? As an **action space**, downstream networks (policy heads, critics, value functions) may expect fixed-size vectors. With this new space, we can skip the manual `jax.nn.one_hot()` conversion that a regular `Discrete` action space would otherwise need. It's a small difference, but it can go a long way.

We can also push it a step further. Envrax's `Discrete` space only supports *uniform* sampling, but in many RL setups (curriculum learning, biased exploration, weighted task selection) you may want some categories to be picked more often than others. We'll add an optional `probs` parameter that lets us supply per-category sampling probabilities to handle this.

Great! That gives us three parameters to work with as we build our `OneHot` space. We'll break its development into four key steps:

1. Defining the class skeleton
2. Implementing `sample()`
3. Implementing `contains()`
4. Implementing `batch()`

### Step 1: Class Skeleton

First up, let's put together the `Space` `dataclass`.

We've already touched on the parameters briefly (`n`, `batch_shape`, and `probs`), but there's one more we need to consider: the `dtype` of the space.

Based on the table we've seen, you might be thinking that an integer `dtype` (e.g., `uint8` or `int32`) is the right fit here. It's a valid choice, but for better compatibility with neural networks, we'd recommend using `float32`. It's the standard for most deep learning workflows and the precision is good enough without needing a `dtype` conversion downstream.

With that in mind, let's put our `Space` together and document it with a suitable docstring:

```python
from dataclasses import dataclass
from typing import Self, Tuple, Type

import chex
import jax
import jax.numpy as jnp

from envrax.spaces import Space


@dataclass(frozen=True)
class OneHot(Space):
    """
    One-hot encoded categorical over `n` classes.

    Parameters
    ----------
    n : int
        Number of categories.
    batch_shape : Tuple[int, ...] (optional)
        Leading shape for batched sampling. Defaults to `()` (unbatched).
    dtype : Type (optional)
        Element dtype. Defaults to `jnp.float32`.
    probs : Tuple[float, ...] | None (optional)
        Per-category sampling probabilities. Must have length `n` and sum
        to 1. When `None` (the default), `sample()` draws uniformly.
    """

    n: int
    batch_shape: Tuple[int, ...] = ()
    dtype: Type = jnp.float32
    probs: Tuple[float, ...] | None = None
```

### Step 2: Implement `sample()`

Next, we'll tackle the `sample()` method. This is pretty simple. All we need to do is create the logic for randomly sampling a one-hot encoded vector.

We can use the `jax.random` module for this. For the `probs` case, we can use `jax.random.choice`, otherwise we fall back to uniform sampling via `jax.random.randint`:

```python
def sample(self, rng: chex.Array) -> chex.Array:
    """
    Sample a random one-hot vector of shape `(*batch_shape, n)`.
    
    Parameters
    ----------
    rng : chex.Array
        JAX PRNG key.

    Returns
    -------
    action : chex.Array
        float32 — One-hot encoded action.
    """
    if self.probs is not None:
        idx = jax.random.choice(
            rng,
            self.n,
            shape=self.batch_shape,
            p=jnp.array(self.probs),
        )
    else:
        idx = jax.random.randint(
            rng,
            shape=self.batch_shape,
            minval=0,
            maxval=self.n,
        )

    return jax.nn.one_hot(idx, self.n, dtype=self.dtype)
```

There are a few things to note here:

- `self.probs is not None` is checked Python-side (at sample-time, not under JIT trace), so this branch is fine — the resulting traced graph is one or the other, not both.
- `jax.nn.one_hot(idx, n)` broadcasts cleanly over any shape — a scalar `idx` produces a vector of length `n`, and a `(k,)` `idx` produces a `(k, n)` matrix. That's exactly what we want for batching with no branching needed! :smile:
- We convert `probs` from a Python tuple to a `jnp.array` inside `sample()` rather than at `__init__` time. This keeps the dataclass itself hashable (tuples are; arrays aren't) without losing JAX compatibility at sample time.

### Step 3: Implement `contains()`

Now for the membership check. This is a little more extensive.

We need to verify that the input matches the expected shape, has exactly one `1` per row, and that all other values are `0`:

```python
def contains(self, x: chex.Array) -> bool:
    """
    Check that `x` is a valid one-hot tensor of the expected shape.
    
    Parameters
    ----------
    x : chex.Array
        Action to validate. Expected to be a one-hot vector.

    Returns
    -------
    valid : bool
        True if `x` is a valid one-hot vector, False otherwise.    
    """
    expected_shape = (*self.batch_shape, self.n)
    if x.shape != expected_shape:
        return False

    is_binary = jnp.all((x == 0) | (x == 1))
    has_one_hot = jnp.all(jnp.sum(x, axis=-1) == 1)
    return bool(is_binary & has_one_hot)
```

This performs three checks: shape, binary-ness, and exactly-one-per-row. If any of them fail, we return `False`.

### Step 4: Implement `batch()`

Lastly, we prepend `n` to `batch_shape` and carry the other parameters through unchanged:

```python
def batch(self, n: int) -> Self:
    """
    Return a `OneHot` with a leading `n` dimension.

    Parameters
    ----------
    n : int
        Batch size.

    Returns
    -------
    batched : OneHot
        An updated `OneHot` instance with
        `batch_shape=(n, *self.batch_shape)`
    """
    return OneHot(
        n=self.n,
        batch_shape=(n, *self.batch_shape),
        dtype=self.dtype,
        probs=self.probs,
    )
```

Easy enough! Since we have a `batch_shape` parameter there's no need to do any crazy space type conversion.

Notice that `probs` is also forwarded as-is - every batched sample uses the same per-category distribution. If you wanted the distribution itself to vary across batch elements, you'd need a different design (e.g., storing `probs` with a leading batch dim too), but for the typical "same env, multiple parallel copies" case, sharing the distribution is what you want.

### Running It

With all methods in place, we can do a quick dummy run to confirm everything is working correctly:

```python
space = OneHot(n=4)
obs = space.sample(jax.random.key(0))  # float32[4], one-hot
space.contains(obs)  # True

batched = space.batch(8)  # OneHot(n=4, batch_shape=(8,))
batched_obs = batched.sample(jax.random.key(0))  # float32[8, 4]
batched.contains(batched_obs)  # True

# Weighted sampling — first category picked ~70% of the time
weighted = OneHot(n=3, probs=(0.7, 0.2, 0.1))
weighted.sample(jax.random.key(0))  # float32[3], biased one-hot
weighted.batch(64).sample(jax.random.key(0))  # float32[64, 3], same distribution per row
```

And there we have it! A new space created and ready for use! :smile:

## Using It on a `JaxEnv`

Custom spaces plug into any `JaxEnv` subclass the same way the built-ins do. Since we built `OneHot` as an action space, it slots into the second generic position (and the second `@property`):

```python
class MyEnv(JaxEnv[Box, OneHot, MyState, EnvConfig]):
    @property
    def observation_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(4,), dtype=jnp.float32)

    @property
    def action_space(self) -> OneHot:
        return OneHot(n=4)
```

And it slots into `VecEnv` just as cleanly:

```python
vec_env = VecEnv(MyEnv(), num_envs=64)
vec_env.action_space  # OneHot(n=4, batch_shape=(64,), dtype=float32)
vec_env.single_action_space  # OneHot(n=4)
```

## Common Pitfalls

Be wary of the following "gotchas":

- **Forgetting `@dataclass(frozen=True)`**. Without `frozen=True`, your space becomes mutable and unhashable — which breaks equality checks (`OneHot(n=4) == OneHot(n=4)` would compare by identity instead of by value), prevents the space from being used as a `dict` key or `set` member, and silently corrupts any code that caches spaces by hash. Every Envrax built-in is `frozen=True`; your custom spaces should match.
- **Not threading `dtype`**. We deliberately exposed `dtype` as a parameter on `OneHot` so users could pick `int32` / `uint8` / `float16` for memory savings or downstream-network compatibility. If you hard-code `jnp.float32` inside `sample()` instead of using `self.dtype`, users lose that flexibility — your space silently ignores their `dtype=...` argument.
- **Using `jax.random.PRNGKey` instead of `chex.Array`**. Functionally identical at runtime, but `chex.Array` is the convention used across every method signature in Envrax (`Space.sample`, `JaxEnv.reset`, every wrapper). Sticking to it keeps your custom space consistent with the API standard so type checkers and IDE hovers behave the same way as the built-ins.
- **Leaking Python-side computation into `sample()`**. The method runs inside JAX traces (`jax.jit`, `jax.vmap`, `jax.lax.scan`), so any `if` or `for` that branches on a *traced* value will raise `ConcretizationTypeError`. The `self.probs is not None` check in our `sample()` method is fine because `self.probs` is a Python attribute on the dataclass — it resolves *before* tracing kicks in, so JAX only ever sees one branch baked into the compiled graph.
- **Using a `jnp.array` instead of a `Tuple` for `probs`-style fields**. We chose `Tuple[float, ...]` for `probs` rather than `jnp.array(...)` for a specific reason: `frozen=True` dataclasses need their fields to be hashable, and JAX arrays aren't (they're mutable buffers underneath). Storing `probs` as a tuple keeps the dataclass valid, and the inline `jnp.array(self.probs)` conversion inside `sample()` is the only place we pay the cost — once per call, not stored.
- **Forgetting to forward optional parameters in `batch()`**. Our `batch()` explicitly passes `dtype=self.dtype` and `probs=self.probs` through to the new instance. If you skip those, the batched copy silently falls back to the defaults — so a `OneHot(n=3, probs=(0.7, 0.2, 0.1))` would suddenly become uniform after `batch(64)`, and your weighted sampling stops working with `VecEnv`. Always forward every instance field.

## Recap

Excellent work! You've just built your first custom Envrax space from scratch! :clap:

Let's recap what we've covered and discuss some key points to consider when building your own custom spaces:

- **Custom `Space` requirements** — every custom space **must** inherit from `envrax.spaces.Space`, use `@dataclass(frozen=True)` for immutable metadata, and implement three abstract methods: `sample(rng)`, `contains(x)`, and `batch(n)`.
- **Designing your parameters** — pick a single required field that defines the space's identity (`n` for us, but it could be `dim` for a simplex, `low`/`high` for a truncated normal, etc.), then layer on optional knobs (`dtype`, `batch_shape`, distribution shape) with sensible defaults. Frozen dataclass fields make this declarative and immutable for free.
- **Writing `sample()`** — keep it pure JAX so it composes with `jax.jit`, `jax.vmap`, and `jax.lax.scan`. Branching on Python-side attributes (like our `if self.probs is not None`) is safe because it resolves before tracing kicks in. This is handy when one space needs to support multiple sampling regimes.
- **Writing `contains()`** — combine the structural checks your space requires (shape, dtype, value bounds, invariants) into a single `bool`. Bail out early on cheap shape mismatches before doing the more expensive elementwise checks to keep things fast.
- **Picking a `batch()` strategy** — we recommend one of two patterns: stay within your own type by tracking `batch_shape` on the instance (clean when the "element" stays the same shape, like `OneHot`), or switch to a different space type when the batched semantics warrant it (the way `Discrete → MultiDiscrete` does).
- **Storing JAX-incompatible config on a frozen dataclass** — frozen dataclasses need every field to be hashable. The best approach is to store the values as a Python `Tuple` (or other immutable container) on the instance, then convert it to a `jnp.array` inside `sample()` at call time. This works for probability vectors (like our example), weight tables, level grids, and anything else you can think of.
- **Forwarding state through `batch()`** — every field you add to the instance has to be explicitly carried through to the new batched copy. Skip even one and the configuration silently disappears the moment `VecEnv` wraps your environment.
