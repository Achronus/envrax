# Multiple Environments

As we've seen, `VecEnv` gives you `N` parallel copies of a *single* environment class, but what if you want to train your agent on *multiple* unique environments?

This is a very common strategy for meta-learning tasks, multi-task training, and when evaluating an agent on multiple environments.

Envrax has built-in support for this via the `MultiEnv` and `MultiVecEnv` classes. Each gives you `M` parallel copies of *different* environment classes. These could be different environments or the same environment but with different observation shapes, action spaces, and configs. The sky's the limit! :smile:

As a rule of thumb, if you want:

1. `N` parallel copies of one environment - use [`VecEnv`](vectorising.md)
2. `M` different environments, one instance for each - use [`MultiEnv`](#multienv)
3. `M` different environments with `N` copies of each (or any mix of [`BatchedEnv`](../../api/env/batched.md) strategies) - use [`MultiVecEnv`](#multivecenv)

## `MultiEnv`

???+ api "API Docs"

    [`envrax.multi_env.MultiEnv`](../../api/env/multi.md#envrax.multi_env.MultiEnv)

`MultiEnv` holds `M` `JaxEnv` instances keyed by environment name and dispatches `reset`/`step` via a Python loop. Each inner environment keeps its own compile cycle (typically via `JitWrapper`) — `MultiEnv` adds no outer `jax.jit` boundary. Reach for [`MultiVecEnv`](#multivecenv) when you need a single jitted dispatch over batched envs instead.

Implementation example :point_down::

```python
import jax
import jax.numpy as jnp

from envrax import MultiEnv
from envrax.wrappers import JitWrapper

multi = MultiEnv([
    JitWrapper(BallEnv()),
    JitWrapper(CartPoleEnv()),
    JitWrapper(BallEnv()),
])
multi.compile()   # trigger XLA compilation for all inner envs

obs, states = multi.reset(jax.random.key(0))
# obs["BallEnv_0"], obs["CartPoleEnv"], obs["BallEnv_1"]

actions = {key: jnp.int32(0) for key in multi.env_keys}
obs, states, rewards, dones, infos = multi.step(states, actions)
```

Some key things worth noting:

1. **Inputs and outputs are `dict`s keyed by environment name**, not Python lists. Different environments may have different observation shapes, action shapes and configs — keying by name keeps everything explicit and easy to look up.
2. **Keys are inferred from each environment's `name`** by default, with `_0`/`_1` suffixes when duplicates appear. Wrappers like `JitWrapper` delegate `name` to the inner environment, so `JitWrapper(BallEnv()).name == "BallEnv"`. Pass a `dict` directly for explicit control.
3. **`reset(rng)` takes one master key**. `MultiEnv` splits it automatically into one sub-key per inner environment so the same master key always produces the same per-environment starts.
4. **`compile()` is a separate step**. `MultiEnv` doesn't pre-warm its inner environments by default. Calling `multi.compile()` walks the fleet and compiles each `JitWrapper`-wrapped environment with a progress bar, so you can measure the setup vs. training costs separately.

### List vs. dict input

Just like `MultiVecEnv`, `MultiEnv` accepts either form:

```python
# Unique names -> bare keys
MultiEnv([BallEnv(), CartPoleEnv()]).env_keys
# ["BallEnv", "CartPoleEnv"]

# Duplicate names -> zero-indexed suffixes
MultiEnv([BallEnv(), BallEnv(), CartPoleEnv()]).env_keys
# ["BallEnv_0", "BallEnv_1", "CartPoleEnv"]

# Explicit dict -> keys used verbatim
MultiEnv({"task_a": BallEnv(), "task_b": BallEnv()}).env_keys
# ["task_a", "task_b"]
```

Iteration order is preserved in both forms.

### `MultiEnv` Attributes

| Item | Description |
| --- | --- |
| `multi.envs` | The `dict` of inner `JaxEnv` instances |
| `multi.env_keys` | Ordered list of environment-type keys |
| `multi.n_envs` | The number of environments (`M`) |
| `multi.observation_spaces` | A `dict` of per-env observation spaces |
| `multi.action_spaces` | A `dict` of per-env action spaces |
| `multi.observation_shapes` | A `dict` of per-env observation shapes (`s.shape`) |
| `multi.action_shapes` | A `dict` of per-env action shapes |
| `multi.observation_sizes` | A `dict` of per-env flat observation sizes (`prod(s.shape)`) |
| `multi.action_sizes` | A `dict` of per-env flat action sizes |
| `multi.observation_dtypes` | A `dict` of per-env observation dtypes |
| `multi.action_dtypes` | A `dict` of per-env action dtypes |

## `MultiVecEnv`

???+ api "API Docs"

    [`envrax.multi_vec_env.MultiVecEnv`](../../api/env/multi.md#envrax.multi_vec_env.MultiVecEnv)

`MultiVecEnv` is the JAX-native sibling of `MultiEnv`. It holds `M` [`BatchedEnv`](../../api/env/batched.md) instances and steps them all together inside a single `jax.jit` boundary, so the cross-environment dispatch loop unrolls at trace time and there's no per-call Python overhead between groups.

Each inner `BatchedEnv` handles its own internal batching however it likes. `VecEnv` is the canonical vmap strategy, but downstream packages can add others (e.g. composite MJX scenes) and slot them into the same `MultiVecEnv` without any changes to envrax.

Implementation example :point_down::

```python
from envrax import MultiVecEnv, VecEnv

multi_vec = MultiVecEnv([
    VecEnv(BallEnv(), num_envs=64),
    VecEnv(CartPoleEnv(), num_envs=64),
])
multi_vec.compile()

obs, states = multi_vec.reset(jax.random.key(0))
# obs["BallEnv"].shape == (64, *ball_obs_shape)
# obs["CartPoleEnv"].shape == (64, *cartpole_obs_shape)

actions = {
    "BallEnv":     jnp.zeros(64, dtype=jnp.int32),
    "CartPoleEnv": jnp.zeros(64, dtype=jnp.int32),
}
obs, states, rewards, dones, infos = multi_vec.step(states, actions)
```

Some key differences from `MultiEnv`:

1. **Inputs and outputs are `dict`s keyed by environment name**, not Python lists. State is a proper JAX pytree — `jax.tree.map`, `jax.tree.leaves`, and friends all work directly on the returned `states` dict.
2. **One `jax.jit` boundary per step**. The Python `for` loop over inner environments runs at trace time, so a single XLA computation dispatches every inner kernel with no per-call Python overhead.
3. **Keys are inferred from each environment's `name`** by default, with `_0`/`_1` suffixes when duplicates appear. Pass a `dict` directly for explicit control.

### List vs. dict input

The `dict` keys in the example above came from `VecEnv.name`, which defaults to the inner `JaxEnv`'s class name. When you supply a list, `MultiVecEnv` derives the keys for you:

```python
# Unique names -> bare keys
MultiVecEnv([
    VecEnv(BallEnv(), 64),
    VecEnv(CartPoleEnv(), 64),
]).env_keys
# ["BallEnv", "CartPoleEnv"]

# Duplicate names -> zero-indexed suffixes
MultiVecEnv([
    VecEnv(BallEnv(), 64),
    VecEnv(BallEnv(), 64),
    VecEnv(CartPoleEnv(), 64),
]).env_keys
# ["BallEnv_0", "BallEnv_1", "CartPoleEnv"]
```

For full control over keys (e.g. task labels like `"task_a"`, `"task_b"`), pass a `dict` directly:

```python
multi_vec = MultiVecEnv({
    "task_a": VecEnv(BallEnv(), 64),
    "task_b": VecEnv(BallEnv(), 64),
})
```

Iteration order is preserved in both forms.

### `MultiVecEnv` Attributes

| Item | Description |
| --- | --- |
| `multi_vec.envs` | The `dict` of inner `BatchedEnv` instances |
| `multi_vec.env_keys` | Ordered list of environment-type keys |
| `multi_vec.n_envs` | The number of distinct environment types (`M`) |
| `multi_vec.total_slots` | Total individual agent slots across all groups |
| `multi_vec.slots_per_env` | A `dict` of per-group slot counts |
| `multi_vec.single_observation_spaces` | A `dict` of per-group unbatched observation spaces |
| `multi_vec.single_action_spaces` | A `dict` of per-group unbatched action spaces |
| `multi_vec.single_observation_shapes` | A `dict` of per-group unbatched observation shapes |
| `multi_vec.single_action_shapes` | A `dict` of per-group unbatched action shapes |
| `multi_vec.single_observation_sizes` | A `dict` of per-group flat unbatched observation sizes (`prod(s.shape)`) |
| `multi_vec.single_action_sizes` | A `dict` of per-group flat unbatched action sizes |
| `multi_vec.single_observation_dtypes` | A `dict` of per-group unbatched observation dtypes |
| `multi_vec.single_action_dtypes` | A `dict` of per-group unbatched action dtypes |

??? tip "Skip the boilerplate with `make_multi` / `make_multi_vec`"
    The [`make_multi`](make.md) and [`make_multi_vec`](make.md) factory methods build these fleets in one call — wrappers, JIT, and all. The class-based form here is the underlying mechanism; the factories are the sugar on top! :muscle:

    Use them whenever you can!

## Additional Methods

Both `MultiEnv` and `MultiVecEnv` share the same dict-keyed surface for finer-grained control.

### Per-Environment Access

Inner environments are directly accessible by key:

```python
inner = multi.envs["BallEnv"]              # MultiEnv     -> JaxEnv
inner = multi_vec.envs["BallEnv"]          # MultiVecEnv  -> BatchedEnv
obs, state = inner.reset(jax.random.key(1))
```

For `MultiVecEnv`, pulling out a single slot's state from a batched environment (e.g. one rollout) uses `slot_state` / `render_slot`:

```python
single_state = multi_vec.slot_state(states, "BallEnv", slot_idx=0)
frame        = multi_vec.render_slot(states, "BallEnv", slot_idx=0)
```

`MultiEnv` doesn't need these — its inner environments aren't batched, so `multi.envs["BallEnv"].render(state)` does the same job directly.

### Padding sizes

Both classes expose `pad_dims()`, which returns the largest flat action and observation sizes across the fleet as an `(action, observation)` tuple:

```python
action, observation = multi_vec.pad_dims()  # e.g. (5, 192)
```

This is useful when you need to `vmap` a single jitted function (or feed one shared policy network) over environments that don't share the same action or observation shapes. Sizes are computed as `prod(space.shape)`, so multi-dim observations are handled correctly.

For `MultiVecEnv` the sizes come from the **unbatched** per-group spaces (i.e., `single_*_sizes`) — that's the dimension a per-environment policy normally uses.

## Common Pitfalls

Using multiple environments at once can be tricky, be mindful of the following "gotchas":

- **Different action dtypes** - if one environment takes `int32` and another takes `float32`, build the actions dict element by element; don't try to `jnp.stack` them.
- **Mismatched keys on `step`** - the `states` and `actions` dicts must have exactly the same keys as `multi.env_keys`. A missing or extra key raises a `ValueError` before the inner step runs.
- **Forgetting `compile()`** - neither class pre-warms its inner environments. Without an explicit `multi.compile()`, your first `step` call will pay the compile cost for *every* inner environment sequentially.

## Recap

To recap:

- `MultiEnv` manages `M` heterogeneous `JaxEnv` instances; `MultiVecEnv` manages `M` heterogeneous [`BatchedEnv`](../../api/env/batched.md) instances (`VecEnv` being the canonical one)
- Both accept either a list (auto-keyed from `env.name` with `_0`/`_1` suffixes on duplicates) or a dict (keys used verbatim)
- Both return dicts keyed by environment name — `MultiVecEnv`'s state is a proper JAX pytree
- Inner environments are accessed via `multi.envs[key]`; for batched slot extraction in `MultiVecEnv`, use `slot_state` / `render_slot`
- `MultiVecEnv` is fully JAX-native — its step runs inside one `jax.jit` boundary with no per-call Python overhead between groups
- Call `.compile()` explicitly — these managers default to deferred compilation

Next up, we'll explore how Envrax's environment registry works so you can use canonical names for building environments instead of classes.

## Next Steps

<div class="grid cards" markdown>

-   :material-book-outline:{ .lg .middle } __Environment Registry__

    ---

    Learn how to use Envrax's environment registry.

    [:octicons-arrow-right-24: Continue to Tutorial 7](registry.md)

</div>
