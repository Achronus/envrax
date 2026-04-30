# Multiple Environments

As we've seen, `VecEnv` gives you `N` parallel copies of a *single* environment class, but what if you want to train your agent on *multiple* unique environments?

This is a very common strategy for meta-learning tasks, multi-task training, and when evaluating an agent on multiple environments.

Envrax has built-in support for this via the `MultiEnv` and `MultiVecEnv` classes. Each gives you `M` parallel copies of *different* env classes. These could be different environments or the same environment but with different observation shapes, action spaces, and configs. The sky's the limit! :smile:

As a rule of thumb, if you want:

1. `N` parallel copies of one environment - use [`VecEnv`](vectorising.md)
2. `M` different environments, one instance for each - use [`MultiEnv`](#multienv)
3. `M` different environments with `N` copies of each - use [`MultiVecEnv`](#multivecenv)

## `MultiEnv`

???+ api "API Docs"

    [`envrax.multi_env.MultiEnv`](../../api/env/multi.md#envrax.multi_env.MultiEnv)

Implementation example :point_down::

```python
import jax
import jax.numpy as jnp

from envrax import MultiEnv
from envrax.wrappers import JitWrapper

# Pass a list of JaxEnv instances directly
multi = MultiEnv([
    JitWrapper(BallEnv()),
    JitWrapper(CartPoleEnv()),
    JitWrapper(BallEnv()),
])
multi.compile()   # trigger XLA compilation for all inner envs

obs_list, states = multi.reset(jax.random.key(0))

actions = [jnp.int32(0) for _ in range(multi.num_envs)]
obs_list, states, rewards, dones, infos = multi.step(states, actions)
```

Some key things worth noting:

1. **Inputs and outputs are Python lists**, not stacked arrays. Different environments may have different observation shapes, action shapes and configs. They cannot be stacked in JAX arrays.
2. **`reset(rng)` takes one master key**. `MultiEnv` splits it automatically into `M` deterministic sub-keys so the same master key always produces the same per-env starts.
3. **`compile()` is a separate step**. `MultiEnv` doesn't pre-warm its inner environments by default. Calling `multi.compile()` walks the fleet and compiles each one with a progress bar, so you can measure the setup vs. training costs separately.

### `MultiEnv` Attributes

| Item | Description |
| --- | --- |
| `multi.envs` | The list of inner `JaxEnv` instances |
| `multi.num_envs` | The number of environments (`M`) |
| `multi.observation_spaces` | A list of per-env observation spaces |
| `multi.action_spaces` | A list of per-env action spaces |
| `multi.class_groups` | A `dict` mapping env class name → list of indices |

## `MultiVecEnv`

???+ api "API Docs"

    [`envrax.multi_vec_env.MultiVecEnv`](../../api/env/multi.md#envrax.multi_vec_env.MultiVecEnv)

Implementation example :point_down::

```python
from envrax import MultiVecEnv, VecEnv

multi_vec = MultiVecEnv([
    VecEnv(BallEnv(), num_envs=64),
    VecEnv(CartPoleEnv(), num_envs=64),
])
multi_vec.compile()

obs_list, states = multi_vec.reset(jax.random.key(0))
# obs_list[0].shape == (64, *ball_obs_shape)
# obs_list[1].shape == (64, *cartpole_obs_shape)

actions = [jnp.zeros(64, dtype=jnp.int32) for _ in range(multi_vec.num_envs)]
obs_list, states, rewards, dones, infos = multi_vec.step(states, actions)
```

This follows the same pattern as `MultiEnv` with a slight difference - each element of the returned lists is batched to `(n_envs, ...)` by its inner `VecEnv`.

### `MultiVecEnv` Attributes

| Item | Description |
| --- | --- |
| `multi_vec.vec_envs` | The list of inner `VecEnv` instances |
| `multi_vec.num_envs` | The number of `VecEnv` groups (`M`) |
| `multi_vec.total_envs` | Total individual environments across all groups (`M × N`) |
| `multi_vec.single_observation_spaces` | A list of per-group unbatched observation spaces |
| `multi_vec.single_action_spaces` | A list of per-group unbatched action spaces |
| `multi_vec.observation_spaces` | A list of per-group batched observation spaces |
| `multi_vec.action_spaces` | A list of per-group batched action spaces |
| `multi_vec.class_groups` | A `dict` mapping inner env class name → list of `VecEnv` indices |

??? tip "Skip the boilerplate with `make_multi` / `make_multi_vec`"
    The [`make_multi`](make.md) and [`make_multi_vec`](make.md) factory methods build these fleets by name in one call - wrappers, JIT, and all. The class-based form here is the underlying mechanism; the factories are the sugar on top. :muscle:

    Use them whenever you can!

## Additional Methods

`MultiEnv` and `MultiVecEnv` share the same extra-API surface for finer-grained control over your fleet of environments.

### Per-Env Access

For targeted environment `resets`/`steps` you can use the utility methods `reset_at()` and `step_at()` to `reset` or `step` a single environment individually:

```python
obs_list[0], states[0] = multi.reset_at(0, jax.random.key(1))
obs, state, reward, done, info = multi.step_at(2, states[2], action)
```

This can be useful for situations like limiting your agents to environment lifetime budgets.

### Class Groups

When your `MultiEnv`/`MultiVecEnv` contains repeat environments (e.g. two `BallEnv` and one `CartPole`), you can group indices by class for downstream same-shape batching:

```python
multi.class_groups
# {"BallEnv": [0, 2], "CartPole": [1]}
```

This is useful if you later want to stack observations for the repeated environment instances into a single batched tensor, perhaps for a policy forward pass or to compute per-task statistics.

## Common Pitfalls

Using multiple environments at once can be tricky, be mindful of the following "gotchas":

- **Different action dtypes** - if `env[0]` takes `int32` and `env[1]` takes `float32`, build the actions list element by element; don't try to `jnp.stack` them.
- **Forgetting `compile()`** - `MultiEnv` and `MultiVecEnv` don't pre-warm their inner environments. Without an explicit `multi.compile()`, your first `step` call will pay the compile cost for *every* env in the fleet sequentially.

## Recap

To recap:

- `MultiEnv` manages `M` heterogeneous `JaxEnv` instances; `MultiVecEnv` manages `M` `VecEnv` groups
- Inputs and outputs are lists because observation shapes can differ across environments
- `reset_at` and `step_at` let you touch a single env without disturbing the rest
- `class_groups` maps class name → indices for downstream same-shape batching
- Call `.compile()` explicitly — these managers default to deferred compilation

Next up, we'll explore how Envrax's environment registry works so you can use canonical names for building environments instead of classes.

## Next Steps

<div class="grid cards" markdown>

-   :material-book-outline:{ .lg .middle } __Environment Registry__

    ---

    Learn how to use Envrax's environment registry.

    [:octicons-arrow-right-24: Continue to Tutorial 7](registry.md)

</div>
