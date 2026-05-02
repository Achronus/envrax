# Creating a Custom Wrapper

You've already used Envrax's built-in wrappers from the [Available Wrappers](../essentials/wrappers.md) tutorial — but what if none of them fit your needs?

Maybe you want to scale your rewards by a constant, add a curriculum learning step, or track a unique statistic across each episode that none of the built-ins cover.

The easiest solution? Building your own wrappers! In this tutorial, we'll walk through how to build both kinds: a **pass-through wrapper** that simply transforms data flowing through `reset`/`step`, and a **stateful wrapper** that needs to remember something between steps.

## Picking a Base Class

???+ api "API Docs"

    - [`envrax.wrappers.base.Wrapper`](../../api/wrappers/base.md#envrax.wrappers.base.Wrapper)
    - [`envrax.wrappers.base.StatefulWrapper`](../../api/wrappers/base.md#envrax.wrappers.base.StatefulWrapper)

Every custom wrapper inherits from one of two base classes:

| Base | When to use | What changes |
| --- | --- | --- |
| `Wrapper` | Pass-through — transforms `obs`/`reward`/`done`, leaves the inner state type unchanged | Nothing |
| `StatefulWrapper` | Stateful — needs to *remember* something between steps | Introduces a new outer state that wraps the inner state |

If you're unsure which one to use, start with `Wrapper`. Then, if you find yourself wanting to store a counter or a running total *across* steps without polluting the inner environment, that's your indicator to transition it to a `StatefulWrapper`.

## Pass-through Wrapper

??? example "Full Code"

    ```python
    from typing import Any, Dict, Tuple

    import chex
    import jax.numpy as jnp

    from envrax.env import ActSpaceT, ConfigT, JaxEnv, ObsSpaceT, StateT
    from envrax.wrappers import Wrapper


    class ScaleReward(Wrapper[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
        """
        Multiplies every reward by a constant `scale`.

        Parameters
        ----------
        env : JaxEnv
            Inner environment to wrap.
        scale : float (optional)
            Scalar multiplier applied to every reward. Default is `1.0`.
        """

        def __init__(
            self,
            env: JaxEnv[ObsSpaceT, ActSpaceT, StateT, ConfigT],
            *,
            scale: float = 1.0,
        ) -> None:
            super().__init__(env)
            self._scale = scale

        def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
            return self._env.reset(rng)

        def step(
            self,
            state: StateT,
            action: chex.Array,
        ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
            obs, new_state, reward, done, info = self._env.step(state, action)
            return obs, new_state, reward * self._scale, done, info
    ```

For this example, we'll build a simple wrapper that multiplies every reward by a constant. We'll call it `ScaleReward`.

We can build it in four key steps:

1. Declaring the class
2. Storing the scale parameter
3. Implementing `reset()`
4. Implementing `step()`

Let's tackle them one at a time.

### Step 1: Declaring the Class

Just like building a `JaxEnv`, we start by subclassing the base class and pinning the generic types. For pass-through wrappers, we leave all four as their defaults:

```python
import chex
from typing import Any, Dict, Tuple

from envrax.env import ActSpaceT, ConfigT, JaxEnv, ObsSpaceT, StateT
from envrax.wrappers import Wrapper


class ScaleReward(Wrapper[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
    ...
```

### Step 2: Storing the Scale Parameter

Next, we add the `__init__` method to accept our `scale` parameter and pass the environment through to the parent class (`Wrapper`):

```python
class ScaleReward(Wrapper[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
    def __init__(
        self,
        env: JaxEnv[ObsSpaceT, ActSpaceT, StateT, ConfigT],
        *,  # (1)
        scale: float = 1.0,
    ) -> None:
        super().__init__(env)
        self._scale = scale
```

1. The `*` marker forces parameters after it (`scale`) to be keyword-only. It's not strictly required, but it's the recommended convention - it stops users from accidentally passing the parameter positionally where `env` is expected, which would break when the wrapper is used through the `make()` methods.

### Step 3: Implementing `reset()`

This one's nice and easy. `ScaleReward` doesn't change anything about the reset path, so we delegate it straight to the inner environment:

```python
def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]:
    return self._env.reset(rng)
```

### Step 4: Implementing `step()`

Lastly, our step method. Like the `reset()` method, we can delegate most of the logic to the inner environment and just unpack the step values.

Then, we scale the reward and return the new value:

```python hl_lines="9"
def step(
    self,
    state: StateT,
    action: chex.Array,
) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]:
    obs, new_state, reward, done, info = self._env.step(state, action)

    # Scale the reward
    scaled_reward = reward * self._scale

    return obs, new_state, scaled_reward, done, info
```

The pattern here is pretty common across most wrappers:

- **Unpack first** to get full access to the inner step's return values
- **Transform what you need** — in this case, just the `reward * self._scale`
- **Pass everything else through unchanged** — `obs`, `new_state`, `done`, and `info` all flow straight through

And that's the whole wrapper! This is far too simple to use in a production setting but gives you an insight into the key fundamentals of wrapper creation.

### Pass-through: Noteworthy Additions

There are a few additional things worth noting:

1. **Type parameters** — For a pure pass-through wrapper, you typically leave all four type parameters the same (`Wrapper[ObsSpaceT, ActSpaceT, StateT, ConfigT]`) so the wrapper inherits whatever the inner environment uses.
2. **`reset` and `step` are both abstract** — you must implement both methods, even if one just delegates (like `reset` does here).
3. **`observation_space` / `action_space` delegate automatically** — you only need to override them if the wrapper changes their shape, dtype, or bounds (e.g., `GrayscaleObservation` drops the channel dim).
4. **Keyword-only parameter** — `scale` uses `*` to force it as a keyword-only parameter. It's not strictly required, but it's the recommended convention since it stops users from accidentally passing the parameter positionally and breaking it for the `make()` methods.

## Stateful Wrapper

??? example "Full Code"

    ```python
    from typing import Any, Dict, Generic, Tuple, TypeVar

    import chex
    import jax.numpy as jnp

    from envrax import EnvState
    from envrax.env import ActSpaceT, ConfigT, JaxEnv, ObsSpaceT
    from envrax.wrappers import StatefulWrapper

    InnerStateT = TypeVar("InnerStateT", bound=EnvState)


    @chex.dataclass
    class MaxRewardState(EnvState, Generic[InnerStateT]):
        """
        State for `MaxReward`.

        Parameters
        ----------
        env_state : InnerStateT
            Forwarded inner environment state (precisely typed).
        max_reward : chex.Array
            Running maximum reward for the current episode (float32).
        """

        env_state: InnerStateT
        max_reward: chex.Array


    class MaxReward(
        StatefulWrapper[
            ObsSpaceT,
            ActSpaceT,
            MaxRewardState[InnerStateT],
            ConfigT,
            InnerStateT,
        ]
    ):
        """
        Tracks the maximum reward seen in the current episode.

        Exposes the running max via `info["max_reward"]` on every `step()` call.

        Parameters
        ----------
        env : JaxEnv
            Inner environment to wrap.
        """

        def __init__(self, env: JaxEnv[ObsSpaceT, ActSpaceT, InnerStateT, ConfigT]) -> None:
            super().__init__(env)

        def reset(
            self, rng: chex.PRNGKey
        ) -> Tuple[chex.Array, MaxRewardState[InnerStateT]]:
            obs, env_state = self._env.reset(rng)
            state = MaxRewardState(
                rng=env_state.rng,
                step=env_state.step,
                done=env_state.done,
                env_state=env_state,
                max_reward=jnp.float32(-jnp.inf),
            )
            return obs, state

        def step(
            self,
            state: MaxRewardState[InnerStateT],
            action: chex.Array,
        ) -> Tuple[
            chex.Array,
            MaxRewardState[InnerStateT],
            chex.Array,
            chex.Array,
            Dict[str, Any],
        ]:
            obs, env_state, reward, done, info = self._env.step(state.env_state, action)

            new_max = jnp.maximum(state.max_reward, reward.astype(jnp.float32))
            reset_max = jnp.where(done, jnp.float32(-jnp.inf), new_max)

            info["max_reward"] = new_max

            new_state = MaxRewardState(
                rng=env_state.rng,
                step=env_state.step,
                done=env_state.done,
                env_state=env_state,
                max_reward=reset_max,
            )
            return obs, new_state, reward, done, info
    ```

Now let's tackle the harder case! :muscle:

We'll build `MaxReward` — a wrapper that tracks the *maximum* reward seen so far in the current episode and exposes it via `info["max_reward"]`.

We can break this down into five key steps:

1. Defining the outer state
2. Declaring the class
3. Setting up `__init__`
4. Implementing `reset()`
5. Implementing `step()`

Like before, we'll tackle them one at a time.

### Step 1: Define the Outer State

Stateful wrappers **must** use a new `@chex.dataclass` that extends `EnvState` and wraps the inner state in an `env_state` field.

We also use `Generic[InnerStateT]` so other wrappers/environments keep their inner state type visible to our custom wrapper:

```python
from typing import Generic, TypeVar

import chex

from envrax import EnvState
from envrax.wrappers import InnerStateT


@chex.dataclass
class MaxRewardState(EnvState, Generic[InnerStateT]):
    env_state: InnerStateT     # forwarded inner state (precisely typed)
    max_reward: chex.Array     # running max reward for this episode (float32)
```

### Step 2: Declaring the Class

Next, we can declare the wrapper itself. Stateful wrappers take **five** generic type parameters instead of four — the extra one (`InnerStateT`) tells the framework what the inner env's state type is:

```python
from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from envrax.env import ActSpaceT, ConfigT, JaxEnv, ObsSpaceT
from envrax.wrappers import StatefulWrapper


class MaxReward(
    StatefulWrapper[
        ObsSpaceT,  # (1)
        ActSpaceT,  # (2)
        MaxRewardState[InnerStateT],  # (3)
        ConfigT,  # (4)
        InnerStateT,  # (5)
    ]
):
    ...
```

1. Observation space type (inherits from inner environment)
2. Action space type (inherits from inner environment)
3. **Outer state type**, pinned to our custom `MaxRewardState` and generic over the inner state
4. Config type (inherits from inner environment)
5. **Inner state type**, allowing the inner environment's state to plug in cleanly

The key things to remember here: keep `InnerStateT` parametric for IDE support, and pin your custom state class to the `StateT` generic position.

### Step 3: Setting up `__init__`

All we do here is accept the environment in `__init__` and delegate it to the parent class:

```python
def __init__(self, env: JaxEnv[ObsSpaceT, ActSpaceT, InnerStateT, ConfigT]) -> None:
    super().__init__(env)
```

No extra parameters or logic needed! If, for example, you wanted a `MaxReward(threshold=0.5)` variant, this is where you'd add it (with the same `*` keyword-only marker we used in `ScaleReward`).

### Step 4: Implementing `reset()`

Now onto the interesting part. On `reset`, we need to:

1. Reset the inner environment to get a fresh inner state
2. Build a new `MaxRewardState` that wraps the inner state and initialises our running max to `-inf`

```python
def reset(
    self, rng: chex.PRNGKey
) -> Tuple[chex.Array, MaxRewardState[InnerStateT]]:
    obs, env_state = self._env.reset(rng)

    state = MaxRewardState(
        rng=env_state.rng,  # (1)
        step=env_state.step,
        done=env_state.done,
        env_state=env_state,  # (2)
        max_reward=jnp.float32(-jnp.inf),
    )

    return obs, state
```

1. **Forward the base fields** (`rng`/`step`/`done`) directly from the inner state. This is what lets `VecEnv`'s auto-reset still see the right `done` flag without having to unwrap our outer state.
2. **Store the inner state** verbatim under `env_state` so we can pass it back to the inner env's `step()` later.

Simple enough!

### Step 5: Implementing `step()`

??? example "Full Method Code"

    ```python
    def step(
        self,
        state: MaxRewardState[InnerStateT],
        action: chex.Array,
    ) -> Tuple[
        chex.Array,
        MaxRewardState[InnerStateT],
        chex.Array,
        chex.Array,
        Dict[str, Any],
    ]:
        obs, env_state, reward, done, info = self._env.step(
            state.env_state, 
            action,
        )

        new_max = jnp.maximum(state.max_reward, reward.astype(jnp.float32))
        reset_max = jnp.where(done, jnp.float32(-jnp.inf), new_max)

        info["max_reward"] = new_max

        new_state = MaxRewardState(
            rng=env_state.rng,
            step=env_state.step,
            done=env_state.done,
            env_state=env_state,
            max_reward=reset_max,
        )
        return obs, new_state, reward, done, info
    ```

Lastly, the *real* fun begins :laughing:.

Firstly, we step the inner environment using the inner state — **not** the outer state, because the inner environment doesn't know our outer state exists:

```python
def step(
    self,
    state: MaxRewardState[InnerStateT],
    action: chex.Array,
) -> Tuple[
    chex.Array,
    MaxRewardState[InnerStateT],
    chex.Array,
    chex.Array,
    Dict[str, Any],
]:
    obs, env_state, reward, done, info = self._env.step(
        state.env_state, 
        action,
    )
    ...
```

Then, we compute the new running max from the latest reward, calculate the reset max for when the episode ends, and store the running max in `info`:

```python
    ...
    new_max = jnp.maximum(state.max_reward, reward.astype(jnp.float32))
    reset_max = jnp.where(done, jnp.float32(-jnp.inf), new_max)

    info["max_reward"] = new_max
    ...
```

Finally, we build a new `MaxRewardState` and return the step values:

```python
    ...
    new_state = MaxRewardState(
        rng=env_state.rng,
        step=env_state.step,
        done=env_state.done,
        env_state=env_state,
        max_reward=reset_max,
    )
    return obs, new_state, reward, done, info
```

That's it! Stateful wrapper done! :sparkles:

### Stateful: Noteworthy Additions

Like our pass-through wrappers, there are a few noteworthy additions to be aware of:

1. **Type parameters** — Stateful wrappers require *five* type parameters (`StatefulWrapper[ObsSpaceT, ActSpaceT, OuterState, ConfigT, InnerState]`). The third one pins your wrapper's outer state type, while the fifth stays parametric so the inner state still has IDE support.
2. **Unwrapping before stepping** — call `self._env.step(state.env_state, action)`, not `self._env.step(state, action)`. The inner environment doesn't know about your outer state.
3. **Copy base fields explicitly** — on both `reset` and `step`, `rng`/`step`/`done` come from the *inner* `env_state`, not the old wrapper state. This is how the auto-reset signal reaches the framework.
4. **Handle episode boundaries** — when `done=True`, your counters should reset. Here we use `jnp.where(done, -inf, new_max)` so the next episode starts fresh. This is the stateful wrapper equivalent of what `VecEnv` already does automatically for the base state.

## Overriding Spaces

Both types of wrapper delegate `observation_space` and `action_space` to the inner environment by default. You only ever need to override them when your wrapper actually *changes* the shape, dtype, or bounds of the data flowing through it.

For example, with the `GrayscaleObservation` wrapper, we drop the channel dim from `(H, W, 3)` down to `(H, W)` and return a new `Box` space:

```python
@property
def observation_space(self) -> Box:
    inner = self._env.observation_space
    h, w = inner.shape[0], inner.shape[1]
    return Box(low=inner.low, high=inner.high, shape=(h, w), dtype=inner.dtype)
```

The `action_space` stays untouched, so there's no need to override it.

The same pattern applies to stateful wrappers — override only the spaces you actually change and let the rest delegate to the inner environment.

## Using Your Wrapper

Once written, custom wrappers integrate seamlessly with the other built-in ones. So, you can use them in the same way as mentioned in the [Available Wrappers](../essentials/wrappers.md#applying-wrappers) tutorial:

```python
from envrax import make

env = make(
    "BallEnv-v0",
    wrappers=[
        MaxReward,                # stateful, non-parameterised — pass the class
        ScaleReward(scale=0.5),   # parameterised — call without env to get a factory
    ],
)

obs, state = env.reset(jax.random.key(0))
obs, state, reward, done, info = env.step(state, action=jnp.int32(0))
print(info["max_reward"])  # Running max tracked by your custom wrapper!
```

No custom integration needed! :smile:

## Common Pitfalls

Like with all our tutorials, be wary of the following "gotchas":

- **Remember to copy `rng`/`step`/`done` to the outer state**. `VecEnv`'s auto-reset reads `state.done` off the outer state, not `state.env_state.done`. If you don't copy the inner `done` flag forward on every step, the outer flag stays `False` forever and your episodes never auto-reset.
- **Calling `self._env.step(state, ...)` instead of `self._env.step(state.env_state, ...)`**. This will raise a `TypeError` because the inner environment doesn't understand your outer state dataclass.
- **Using Python scalars for episode-lifetime counters**. These become static at trace time and will break your code. Always use `jnp.int32(value)` / `jnp.float32(value)` instead.
- **Skipping `Generic[InnerStateT]`**. The wrapper still works, but inner wrappers/environments lose type safety on `env_state` and IDE autocomplete drops back to `EnvState`.
- **Skipping the `*` keyword-only marker on parameterised args**. If a wrapper parameter is positional, users can accidentally pass it where `env` is expected — breaking the wrapper when it's used through the `make()` methods. Always use `def __init__(self, env, *, param=...)` when adding new parameters to your wrappers.

## Recap

And that's a wrap! :clap: (pun intended :wink:)

To recap:

- `Wrapper` is for pass-through behaviour; `StatefulWrapper` is for when you need to remember something across steps.
- Stateful wrapper states use `Generic[InnerStateT]`, hold an `env_state` field, and forward `rng`/`step`/`done` from the inner state.
- Use keyword-only parameters on `__init__` so users can call your wrapper safely through the `make()` methods.
- Override `observation_space`/`action_space` only when you need to modify them.
- Always reset stateful counters on `done=True` so the next episode starts fresh.
