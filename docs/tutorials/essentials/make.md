# Make Methods

Once your environments are [registered](registry.md) to the registry, you can start using them with the `make` factory methods.

## Methods Overview

Here's a list of them and when to use them:

| Factory | Returns | Use for |
| --- | --- | --- |
| `make(name)` | `JaxEnv` | A single environment |
| `make_vec(name, n_envs)` | `VecEnv` | A batched environment (for parallel rollouts) |
| `make_multi(names)` | `MultiEnv` | For single use heterogeneous environments |
| `make_multi_vec(names, n_envs)` | `MultiVecEnv` | For batched heterogeneous environments |

### Arguments

All four mostly share the same keyword arguments, so you only need to learn them once! These include:

| Keyword | Type | Default | Description |
| --- | --- | --- | --- |
| `wrappers` | `List[WrapperType]` or `None` | `None` | The wrappers to apply to the environment(s). For the multi variants, the same pipeline is applied to *every* environment in the list. |
| `jit_compile` | `bool` | `True` | Wraps the environment(s) in a `JitWrapper` to enable the XLA compilation cache. |
| `pre_warm` | `bool` | `True` for `make` / `make_vec`; `False` for `make_multi` / `make_multi_vec` | When `jit_compile=True`, runs a dummy `reset` + `step` immediately to trigger XLA compilation. Otherwise, compilation is deferred to the first call or an explicit `.compile()` call. |
| `cache_dir` | `Path` or `str` or `None` | `~/.cache/envrax/xla_cache` | The directory for the persistent XLA compilation cache. Pass `None` to disable. |

The single-env make methods (`make`, `make_vec`) take an additional `config` keyword:

| Keyword | Type | Default | Description |
| --- | --- | --- | --- |
| `config` | `EnvConfig` or `None` | `None` | Overrides the registered default config for this single environment. |

The vector make methods add one additional positional parameter:

| Method | Extra Parameter | Description |
| --- | --- | --- |
| `make_vec(name, n_envs, ...)` | `n_envs: int` | The number of parallel copies inside the returned `VecEnv`. |
| `make_multi_vec(names, n_envs, ...)` | `n_envs: int` | The number of parallel copies *per* environment in the returned `MultiVecEnv`. |

## `make()`

???+ api "API Docs"

    [`envrax.make.make`](../../api/make.md#envrax.make.make)

Use this method for creating single environments. It returns a wrapped environment ready to use.

Implementation example :point_down::

```python
import envrax

env = envrax.make("BallEnv-v0")
# env: JitWrapper<BallEnv> (JIT-compiled by default)
```

Override the default config with the `config` parameter:

```python
env = envrax.make(
    "BallEnv-v0",
    config=BallConfig(max_steps=10_000),
)
```

Or, apply wrappers in one shot (no `functools.partial` needed!):

```python
from envrax.wrappers import NormalizeObservation, ClipReward, FrameStackObservation

env = envrax.make(
    "BallEnv-v0",
    wrappers=[
        NormalizeObservation,
        FrameStackObservation(n_stack=4),   # parameterised — call without env
        ClipReward,
    ],
)
```

We'll discuss the full list of [Available Wrappers](wrappers.md) Envrax offers in the next tutorial.

For JIT-compilation, `make()` wraps the environment in a `JitWrapper` by default and pre-warms the XLA cache on construction.

You can opt-out of this using the `jit_compile` and `pre_warm` parameters:

```python
env = envrax.make(
    "BallEnv-v0", 
    jit_compile=False, # no JitWrapper
    pre_warm=False, # Disable auto-compilation
)
```

Disable `jit_compile` when you need a Python-side env (debugging, evaluation with non-JIT wrappers like `RecordVideo`) and disable `pre_warm` when you'd rather pay the compilation cost lazily on the first environment call.

## `make_vec()`

???+ api "API Docs"

    [`envrax.make.make_vec`](../../api/make.md#envrax.make.make_vec)

Use this method for creating single batched environments. It returns a `VecEnv` ready to use.

This follows the same principles as `make()` but requires the `n_envs` parameter to make multiple copies of the environment:

```python
vec_env = envrax.make_vec("BallEnv-v0", n_envs=64)
obs, states = vec_env.reset(jax.random.key(0)) # obs: [64, ...]
```

Wrappers are applied to the *inner* env first (the per-env transformations), then `VecEnv` wraps the wrapped env. See [Vectorising](vectorising.md#using-wrappers) for the rationale.

## `make_multi()`

???+ api "API Docs"

    [`envrax.make.make_multi`](../../api/make.md#envrax.make.make_multi)

Use this method for creating a list of heterogeneous environments. It returns a `MultiEnv` that manages the environments using the same API as normal with a few additional methods.

This is useful for meta-learning, multi-task training, or evaluation suites that span multiple environments.

Like `make_vec()`, it follows the same principles as `make()` but takes a *list* of registered names instead of a single name:

```python
multi = envrax.make_multi(["BallEnv-v0", "CartPole-v0"])
```

Each environment is constructed with its **registered default config**. If you need per-env overrides, register the variants ahead of time (e.g. `BallEnv-easy-v0`, `BallEnv-hard-v0`) or compose them manually using the `MultiEnv` class instead:

```python
from envrax import MultiEnv

multi = MultiEnv([
    envrax.make("BallEnv-v0", config=BallConfig(max_steps=200)),
    envrax.make("CartPole-v0"),
])
```

Wrappers apply to every environment in the list. The same pipeline must be compatible with every environment's observation and action space; if a subset of the environments needs different wrappers, build two (or more) separate `MultiEnv`s instead.

Unlike `make()`, the `pre_warm` parameter defaults to `False` here. JIT wrapping still happens at construction, but XLA compilation is deferred so you don't pay the cost `N` times in a row. Trigger it explicitly as a separate setup phase using the `compile()` method:

```python
multi = envrax.make_multi(["BallEnv-v0", "CartPole-v0"])
multi.compile()  # warms every env together (with a progress bar)
```

See [Multiple Environments](multi-env.md) for the full `MultiEnv` API.

## `make_multi_vec()`

???+ api "API Docs"

    [`envrax.make.make_multi_vec`](../../api/make.md#envrax.make.make_multi_vec)

Use this method for creating a list of heterogeneous *batched* environments. It returns a `MultiVecEnv` where each entry is already vectorised across `n_envs` parallel copies.

This follows the same principles as `make_multi()` but requires the `n_envs` parameter to make multiple copies of *each* environment:

```python
multi_vec = envrax.make_multi_vec(
    ["BallEnv-v0", "CartPole-v0"],
    n_envs=64,
)
```

As with `make_multi()`, there is no `config` parameter — each environment uses its registered default.

Wrappers are applied to the *inner* env first (the per-env transformations), then `VecEnv` wraps each one following the same nesting order as `make_vec()`. It has the same compatibility constraint as `make_multi()`: every env in the list must accept the same wrapper pipeline.

Again, like `make_multi()`, `pre_warm` defaults to `False`. Call `multi_vec.compile()` as a separate setup phase to warm every `VecEnv` together:

```python
multi_vec = envrax.make_multi_vec(["BallEnv-v0", "CartPole-v0"], n_envs=64)
multi_vec.compile()
```

Refer to the [Multiple Environments](multi-env.md) tutorial for the full `MultiVecEnv` API.

## Recap

To recap:

- `make()`, `make_vec()`, `make_multi()`, `make_multi_vec()` all use canonical ID lookups to get environments from the registry
- All four share `wrappers`, `jit_compile`, `pre_warm`, and `cache_dir` keyword arguments
- `make()` and `make_vec()` accept a `config` argument for per-env overrides
- `make_multi()` and `make_multi_vec()` use each environment's registered default `config`
- Wrappers compose innermost-first and parameterised ones are called without `env` to defer binding
- The wrapper pipeline for `make_multi()` / `make_multi_vec()` must be compatible with every env in the list — if not, split them into multiple `MultiEnv`s
- `jit_compile=False` opts out of `JitWrapper`; `pre_warm=False` defers XLA compilation
- `make_multi` methods default to `pre_warm=False` requiring a separate `.compile()` call

Next up, we'll explore the available wrappers Envrax has to offer!

## Next Steps

<div class="grid cards" markdown>

-   :material-layers-outline:{ .lg .middle } __Available Wrappers__

    ---

    Browse the built-in wrapper catalogue.

    [:octicons-arrow-right-24: Continue to Tutorial 9](wrappers.md)

</div>
