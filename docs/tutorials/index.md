# Tutorials

Welcome to the tutorials! This section teaches you how to build, run, and extend Envrax environments for all your RL environment needs.

If you are new to Envrax, we highly recommend working through the [Essentials](#essentials) to get comfortable with the basics. Each tutorial builds on the other to help you gain a better understanding of each concept and become an Envrax master in no time! :wink:

If you are already familiar with the basics, check out the [Advanced](#advanced) section for more task-focused walkthroughs.

??? info "Already an Expert?"
    Then what are you still doing here?! :face_with_raised_eyebrow: Get out there and build some environments! :rocket:

## Prerequisites

These tutorials assume:

- Python 3.13+ is installed with Envrax. If not, refer to :point_right: [Getting Started](../starting/index.md)
- Basic familiarity with [JAX [:material-arrow-right-bottom:]](https://docs.jax.dev/en/latest/) — particularly `jax.jit`, `jax.vmap`, and `jax.random`
- Comfort with Python [dataclasses [:material-arrow-right-bottom:]](https://docs.python.org/latest/library/dataclasses.html) and [generics [:material-arrow-right-bottom:]](https://typing.python.org/en/latest/reference/generics.html)
- Basic familiarity with chex [dataclasses [:material-arrow-right-bottom:]](https://chex.readthedocs.io/en/latest/api.html#dataclasses)

If any of that is unfamiliar, we highly recommend referring to the linked resources first and come back once comfortable. You'll get more out of the tutorials that way!

## Essentials

??? note "New to Envrax?"

    Start here! :point_down:

Each tutorial is a short, self-contained guide that includes a runnable code snippet to help get you familiar with the basics.

We recommend completing the tutorials in order below to get the most out of this tutorial series.

| # | Tutorial | Teaches |
| - | --- | --- |
| 1 | [Environment State](essentials/state.md) | What state is, extending `EnvState`, threading `rng`, and managing per-episode fields |
| 2 | [Spaces](essentials/spaces.md) | What spaces are and using contracts like `Box`, `Discrete`, and `MultiDiscrete` to describe your environment |
| 3 | [Environment Configuration](essentials/configuration.md) | Extending `EnvConfig` with your own static fields |
| 4 | [Your First Environment](essentials/first-env.md) | Subclassing `JaxEnv`, implement the primary methods `reset` and `step`, and how to use it |
| 5 | [Vectorising with `VecEnv`](essentials/vectorising.md) | Running `N` parallel copies via `jax.vmap` |
| 6 | [Multiple Environments](essentials/multi-env.md) | Managing `M` heterogeneous envs with `MultiEnv` / `MultiVecEnv` |
| 7 | [Environment Registry](essentials/registry.md) | Registering envs by name and discovering registered specs |
| 8 | [Make Methods](essentials/make.md) | Constructing registered envs with `make()`, `make_vec()`, `make_multi()`, and `make_multi_vec()` |
| 9 | [Available Wrappers](essentials/wrappers.md) | A catalogue of built-in wrappers for transforming observations, rewards, and episode statistics |
| 10 | [Rendering](essentials/rendering.md) | Implementing `render(state)` for visual inspection |

## Advanced

??? note "Familiar with the basics?"
    Use these tutorials to level up further! :muscle:

Each tutorial is a task-focused guide for specific features. These can be read in any order and work independently.

| # | Tutorial | Description |
| - | --- | --- |
| 1 | [Recording Video](advanced/recording-video.md) | Save episode rollouts as MP4 with `RecordVideo` |
| 2 | [Creating a Custom Wrapper](advanced/custom-wrapper.md) | Build your own pass-through and stateful wrapper |
| 3 | [Creating a Custom Space](advanced/custom-spaces.md) | Subclass `Space` to support data shapes Envrax doesn't ship — simplex, one-hot, bitstring, etc. |
| 4 | [Debugging JIT'd Environments](advanced/debugging.md) | Trace-aware logging, in-JIT assertions, and the `info`-channel pattern for observing what your env actually does |
