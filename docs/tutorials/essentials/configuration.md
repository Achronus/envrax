# Environment Configuration

Before we wire `state` and `spaces` into a working `JaxEnv`, there's one more piece to introduce: how to handle the environment's **core**/**unique** properties. This is where `EnvConfig` comes in.

Every `JaxEnv` holds one under the `self.config` property. Here's what the base looks like:

```python
@chex.dataclass
class EnvConfig:
    max_steps: int = 1000
```

One field that defines the default episode length of the environment. Nice and simple! :wink:

`EnvConfig` is designed as **static** data that is set once at construction and never changed through the episode. If your environment has gravity, reward scales, difficulty modes, or level seeds; this is where to put them. :muscle:

## EnvConfig vs. EnvState

Now you may be wondering: "Can't everything just live in `EnvState`?" While technically true, the key distinction is in how JAX handles **static** vs. **traceable** data. As we mentioned in our earlier tutorials, we need to be careful not to mix static data with traceable data.

> **Traceable** values act as runtime data, allowing them to be changed during each function call without triggering a JIT-compile. **Static** values, on the other hand, need to be re-traced and re-compiled whenever they change.

As a rule of thumb:

1. If an item is fixed for the whole episode, needs to be known at construction time, is a Python scalar or has a static shape, it **goes in `EnvConfig`**.
2. If an item changes during the episode, only needs to be known at runtime, is a JAX array or JAX compatible, it **goes in `EnvState`**.

**Remember**: you should only ever need to set the config once at environment creation. Otherwise, JIT will silently use the old cached values and break your training loop without warning.

## Extending `EnvConfig`

???+ api "API Docs"

    [`envrax.env.EnvConfig`](../../api/env/base.md#envrax.env.EnvConfig)

Now let's look at how we can extend `EnvConfig`. Just like `EnvState`, we use the `@chex.dataclass` decorator and subclass from the parent class (`EnvConfig`). Then, just add the fields we want:

```python
import chex
from envrax import EnvConfig

@chex.dataclass
class BallConfig(EnvConfig):
    friction: float = 0.98
    reward_scale: float = 1.0

    # max_steps: int = 1000 # (1)
```

1. Included for free! Simply uncomment it and change `1000` to increase the default episode length of this environment

## Pinning the Config to Your Environment

When you build a `JaxEnv` subclass, you'll need to *pin* your custom config (`BallConfig`) as the fourth generic parameter:

```python
class BallEnv(JaxEnv[Box, Discrete, BallState, BallConfig]):
    ...
```

We'll discuss this in more detail in the next tutorial ([Your First Environment](first-env.md)).

## Recap

To recap:

- `EnvConfig` holds static per-env data; `EnvState` holds dynamic per-episode data
- Extend `EnvConfig` with `@chex.dataclass` and add fields with Python scalar defaults
- Pin your custom config as the 4th generic parameter on `JaxEnv`

## Next Steps

You've now seen all three foundational pieces — state, spaces, and config. Time to wire them into a working environment!

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Your First Environment__

    ---

    Subclass `JaxEnv`, implement `reset` and `step`, and use `BallConfig` to drive the dynamics.

    [:octicons-arrow-right-24: Continue to Tutorial 4](first-env.md)

</div>
