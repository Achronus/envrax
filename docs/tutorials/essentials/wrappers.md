# Available Wrappers

Sometimes when doing RL experiments you need some minor differences for a specific environment. Maybe you want its observation state to be in a different shape or its rewards to automatically be bounded between `0` and `1`.

Wrappers are a simple way to make these types of changes and are useful for extending, enhancing, or updating a portion of an environment without modifying its source directly.

They take an inner `JaxEnv`, change one or more of its inputs/outputs (`observations`, `rewards`, `state`, `done` flag, `info` metadata), and expose the same `reset`/`step` interface so everything downstream - the `VecEnv` classes, `make()` methods, your training loop - keeps working without any changes.

This tutorial covers the pre-built wrappers provided with Envrax and how to use them in your projects.

??? tip "Want to create your own?"

     See the :point_right: [Creating a Custom Wrapper](../advanced/custom-wrapper.md) :point_left: tutorial.

## Types of Wrappers

???+ api "API Docs"

    - [`envrax.wrappers.base.Wrapper`](../../api/wrappers/base.md#envrax.wrappers.base.Wrapper)
    - [`envrax.wrappers.base.StatefulWrapper`](../../api/wrappers/base.md#envrax.wrappers.base.StatefulWrapper)

Every Envrax wrapper falls into one of two categories - **pass-through** (stateless) and **stateful**.

The main difference is whether the wrapper introduces its own state alongside the inner environment's. If it does, it's classed as a **stateful** wrapper.

Simply put, **stateful** wrappers need to *remember* something across timesteps such as a rolling frame buffer, or an episode total. This ensures that the environments are still compatible with JAX's transforms (`jit`, `vmap`, `scan`). We'll discuss this in more depth shortly.

First though, we'll explore the simpler variant of the two: pass-through wrappers!

## Pass-through Wrappers

???+ api "API Docs"

    [`envrax.wrappers.passthrough`](../../api/wrappers/passthrough.md)

These wrappers don't introduce any new state. Instead, they just transform the desired inputs/outputs and flow through the `reset`/`step` methods like normal.

Here's a quick overview of available pass-through wrappers:

| Wrapper | Input obs | Output obs | Description |
| --- | --- | --- | --- |
| [`JitWrapper`](#jitwrapper) | any | same | Compiles `reset` + `step` with `jax.jit`; caches kernels to disk |
| [`GrayscaleObservation`](#grayscaleobservation) | `uint8[H, W, 3]` | `uint8[H, W]` | NTSC luminance conversion |
| [`ResizeObservation(h, w)`](#resizeobservation) | `uint8[H, W]` or `uint8[H, W, C]` | `uint8[h, w]` or `uint8[h, w, C]` | Bilinear resize (default `84×84`) |
| [`NormalizeObservation`](#normalizeobservation) | `uint8[...]` | `float32[...]` in `[0, 1]` | Divide by 255 |
| [`ClipReward`](#clipreward) | any reward | `float32 ∈ {−1, 0, +1}` | Sign clipping |
| [`ExpandDims`](#expanddims) | any | same | Adds trailing size-1 dim to `reward` and `done` |
| [`EpisodeDiscount`](#episodediscount) | any | same | Converts `done` bool to `float32` discount (`1.0` / `0.0`) |
| [`RecordVideo`](#recordvideo) | any | same | Saves episode frames to MP4 (not JIT-compatible) |

We'll dig into each one below.

### `JitWrapper`

???+ api "API Docs"

    [`envrax.wrappers.jit_wrapper.JitWrapper`](../../api/wrappers/passthrough.md#envrax.wrappers.jit_wrapper.JitWrapper)

This wrapper JIT-compiles the `reset` and `step` steps and caches the resulting XLA executables to disk.

You'll find it applied automatically by the [`make()`](make.md) methods when setting `jit_compile=True`.

You should rarely need to construct it manually, but if you do, here's an example:

```python
from envrax.wrappers import JitWrapper

env = JitWrapper(BallEnv())  # (1)
```

1. `pre_warm=True` by default

It also exposes a `compile()` method so you can trigger the XLA compilation manually. This is useful when you've constructed the wrapper with `pre_warm=False` and want to defer the compilation cost to a separate setup phase:

```python
env = JitWrapper(BallEnv(), pre_warm=False)
# ...do other setup...
env.compile()  # warms the XLA cache for this env
```

It's worth noting, `compile()` is safe to call multiple times. Thanks to caching, subsequent calls should be near-instant on wall-clock time making failed runs operate that little bit smoother!

### `GrayscaleObservation`

???+ api "API Docs"

    [`envrax.wrappers.grayscale.GrayscaleObservation`](../../api/wrappers/passthrough.md#envrax.wrappers.grayscale.GrayscaleObservation)

??? warning "Combining with Other Wrappers"

    When using this with the [`NormalizeObservation`](#normalizeobservation) wrapper, you should always apply this *before* it. Grayscale expects `uint8` values, not `float`.

This wrapper converts an RGB observation to grayscale using NTSC luminance weights (`0.299 R + 0.587 G + 0.114 B`).

| Input obs | Output obs |
| --- | --- |
| `uint8[H, W, 3]` | `uint8[H, W]` |

```python
from envrax.wrappers import GrayscaleObservation

env = GrayscaleObservation(BallEnv())
```

### `ResizeObservation`

???+ api "API Docs"

    [`envrax.wrappers.resize.ResizeObservation`](../../api/wrappers/passthrough.md#envrax.wrappers.resize.ResizeObservation)

This wrapper performs bilinear-resizing on 2-D or 3-D `uint8` observations to a target height and width `(h, w)`. The channel dimension (`C`) is preserved.

| Input obs | Output obs |
| --- | --- |
| `uint8[H, W]` | `uint8[h, w]` |
| `uint8[H, W, C]` | `uint8[h, w, C]` |

```python
from envrax.wrappers import ResizeObservation

env = ResizeObservation(GrayscaleObservation(BallEnv()), h=84, w=84)
```

### `NormalizeObservation`

???+ api "API Docs"

    [`envrax.wrappers.normalize_obs.NormalizeObservation`](../../api/wrappers/passthrough.md#envrax.wrappers.normalize_obs.NormalizeObservation)

??? warning "Combining with Other Wrappers"

    When using this with shape-transform wrappers like [`GrayscaleObservation`](#grayscaleobservation) and [`ResizeObservation`](#resizeobservation), you should always apply this *after* them. Those wrappers expect `uint8` values, not `float`.

This wrapper divides `uint8` observations by `255` and casts them to `float32`, normalizing their values between the range of `[0, 1]`.

| Input obs | Output obs |
| --- | --- |
| `uint8[...]` | `float32[...]` in `[0, 1]` |

```python
from envrax.wrappers import NormalizeObservation

env = NormalizeObservation(BallEnv())
```

### `ClipReward`

???+ api "API Docs"

    [`envrax.wrappers.clip_reward.ClipReward`](../../api/wrappers/passthrough.md#envrax.wrappers.clip_reward.ClipReward)

This wrapper sign-clips rewards to `{-1, 0, +1}`. It's useful as a stabilisation step when reward magnitudes can vary wildly between episodes or across environments.

```python
from envrax.wrappers import ClipReward

env = ClipReward(BallEnv())
```

### `EpisodeDiscount`

???+ api "API Docs"

    [`envrax.wrappers.discount.EpisodeDiscount`](../../api/wrappers/passthrough.md#envrax.wrappers.discount.EpisodeDiscount)

This wrapper converts the `done` boolean to a `float32` discount factor (`1.0` when not done, `0.0` when done). Useful for value bootstrapping where you want `value(s') * discount` to zero out at terminal states.

```python
from envrax.wrappers import EpisodeDiscount

env = EpisodeDiscount(BallEnv())
```

### `ExpandDims`

???+ api "API Docs"

    [`envrax.wrappers.expand_dims.ExpandDims`](../../api/wrappers/passthrough.md#envrax.wrappers.expand_dims.ExpandDims)

This wrapper adds a trailing size-1 dimension to `reward` and `done` so they broadcast cleanly against batched value heads.

```python
from envrax.wrappers import ExpandDims

env = ExpandDims(BallEnv())
# step returns reward.shape == (1,), done.shape == (1,)
```

### `RecordVideo`

???+ api "API Docs"

    [`envrax.wrappers.record_video.RecordVideo`](../../api/wrappers/passthrough.md#envrax.wrappers.record_video.RecordVideo)

??? warning "JIT and vmap Incompatibility"

    `RecordVideo` is **not JIT or vmap compatible** because it writes files Python-side. Use it for evaluation, logging, or training visualisation **only**. Never use it inside `jax.jit`/`jax.vmap`.

    Calling `reset` or `step` inside any `jax.jit`, `jax.vmap`, or `jax.lax.scan` boundary raises a `RuntimeError`.

    The wrapped environment must also implement a `render(state)` method. Otherwise, `RecordVideo` will raise a `TypeError` at construction.

This wrapper saves episode frames to MP4 via `imageio` and can be customized based on three optional trigger controls.

```python
from envrax.wrappers import RecordVideo

env = RecordVideo(BallEnv(), output_dir="runs/eval", fps=30)
```

Output files are stored in `<output_dir>/episode_<NNNN>.mp4`. The wrapper requires `imageio` with the `ffmpeg` plugin, which you can install via:

=== "uv"
    ```bash title=""
    uv add "imageio[ffmpeg]"
    ```

=== "pip"
    ```bash title=""
    pip install "imageio[ffmpeg]"
    ```

=== "poetry"
    ```bash title=""
    poetry add "imageio[ffmpeg]"
    ```

#### Trigger Controls

To make this wrapper more flexible, you can configure specific triggers based on your requirements to control when recording is active. If no triggers are provided, **every** episode is recorded.

Here are your options:

- `episode_trigger: Callable[[int], bool]` — fires at each `reset()` with the current episode index
- `step_trigger: Callable[[int], bool]` — fires at each `step()` with the global step count, and starts recording mid-episode
- `recording_trigger: Callable[[], bool]` — zero-arg callable checked at each `reset()`, useful for external control via a custom flag

##### Episode Trigger

Use this when you want to record on a regular cadence (e.g., every `N`th episode):

```python
env = RecordVideo(
    BallEnv(),
    output_dir="runs/eval",
    episode_trigger=lambda ep: ep % 100 == 0,  # every 100th episode
)
```

##### Step Trigger

Use this when you want to start recording mid-episode after a global step threshold. It can also be handy for skipping the first `N` warmup steps and starting recording afterwards:

```python
env = RecordVideo(
    BallEnv(),
    output_dir="runs/eval",
    step_trigger=lambda s: s >= 10_000,  # start once total steps ≥ 10k
)
```

Once the trigger fires, recording continues until that episode ends.

##### Recording Trigger

Use this when an external system (e.g. a meta-learning loop or evaluation harness) controls when recording is active via a custom flag:

```python
should_record = False

env = RecordVideo(
    BallEnv(),
    output_dir="runs/eval",
    recording_trigger=lambda: should_record,
)

# Toggle from outside the env:
should_record = True
```

##### Combining Triggers

If you want, you can mix and match your triggers and combine them together! If any one of them returns `True`, recording will fire:

```python
env = RecordVideo(
    BallEnv(),
    output_dir="runs/eval",
    episode_trigger=lambda ep: ep % 100 == 0,    # every 100th episode
    step_trigger=lambda s: s >= 10_000,          # OR after 10k steps
    recording_trigger=lambda: should_record,     # OR when flag is set
)
```

## Stateful Wrappers

???+ api "API Docs"

    [`envrax.wrappers.stateful`](../../api/wrappers/stateful.md)

These wrappers introduce their own outer state so that they can *remember* the information they need to carry across timesteps.

### `FrameStackObservation`

???+ api "API Docs"

    [`envrax.wrappers.frame_stack.FrameStackObservation`](../../api/wrappers/stateful.md#envrax.wrappers.frame_stack.FrameStackObservation)

This wrapper maintains a sliding window of the last `n_stack` observations in a rolling manner.

This is useful when you need your agent to perceive motion.

| Input obs | Output obs |
| --- | --- |
| `uint8[H, W]` | `uint8[H, W, n_stack]` |

```python
from envrax.wrappers import FrameStackObservation

env = FrameStackObservation(
    ResizeObservation(GrayscaleObservation(BallEnv()), h=84, w=84),
    n_stack=4,
)
```

### `RecordEpisodeStatistics`

???+ api "API Docs"

    [`envrax.wrappers.record_episode_statistics.RecordEpisodeStatistics`](../../api/wrappers/stateful.md#envrax.wrappers.record_episode_statistics.RecordEpisodeStatistics)

This wrapper tracks the cumulative return and step count of each episode.

It adds an `episode` entry to the `info` metadata on **every** `step()`, populated only when `done=True`, providing an episode return value (`r`) and episode length (`l`). Format:

```python
info["episode"] = {
    "r": float32,  # cumulative episode return — 0.0 except on done=True
    "l": int32,    # episode length in timesteps — 0 except on done=True
}
```

These are useful for logging episodic metrics dynamically without having to manually create them yourself!

```python
from envrax.wrappers import RecordEpisodeStatistics

env = RecordEpisodeStatistics(BallEnv())
_, state = env.reset(key)
_, state, _, done, info = env.step(state, action)

# At the end of an episode:
info["episode"]   # {"r": episode_return: 1.0, "l": episode_length: 35}
```

## Applying Wrappers

We can apply wrappers through two methods:

- Using the built in `make()` methods
- Manually through class instances

### Using `make()` methods

The easiest way to apply wrappers is through the [`make()`](make.md) methods.

Simply provide them as class types (just the class name) or their full class with custom parameters (without `env=`) and the selected `make` method will do the rest for you!

```python
from envrax import make
from envrax.wrappers import (
    ClipReward,
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)

env = make(
    "BallEnv-v0",
    wrappers=[
        GrayscaleObservation,
        ResizeObservation(h=84, w=84),
        FrameStackObservation(n_stack=4),
        ClipReward,
    ],
)
```

There are a few things to consider when using this approach:

1. **Order matters.**

    Wrappers apply innermost-first, so the list operates in a top-down approach:

    > Grayscale :material-arrow-right: Resize :material-arrow-right: Frame-stack :material-arrow-right: Clip reward

    Swapping the order will produce different results.

2. **Parameterised wrappers must be called without `env`.**

    Under the hood, parameterised wrappers (e.g., `ResizeObservation(h=84, w=84)`) return a `_WrapperFactory` that the `make()` method finishes binding to the base environment automatically once it's constructed.

    There's no need for `functools.partial`! :muscle:

### Manually

You can also apply the wrappers manually, without a `make()` method, using direct calls like so:

```python
env = GrayscaleObservation(BallEnv())
env = ResizeObservation(env, h=84, w=84)
env = FrameStackObservation(env, n_stack=4)
env = ClipReward(env)
```

This can be useful in unit tests or when you want to construct a wrapper chain yourself.

## Accessing the Inner Environment

Every wrapper exposes an `env.unwrapped` field to give you access to the innermost (initial/base) environment.

For example, if we wrapped our `BallEnv` and wanted to check its base instance instead of its `ClipReward` variant, we could grab it using this field:

```python
wrapped = ClipReward(BallEnv())
type(wrapped.unwrapped).__name__   # 'BallEnv'
```

This behaviour holds no matter how many wrappers you apply!

??? note "Obs and Action Space Behaviour"

    `observation_space`/`action_space` delegate to the inner environment by default. 
    
    Wrappers only override them when they *change* the space. For example, the `GrayscaleObservation` wrapper drops the channel dimension, so the `observation_space` is modified.

## Common Pipelines

**Atari-style image preprocessing:**

```python
wrappers=[
    GrayscaleObservation,
    ResizeObservation(h=84, w=84),
    NormalizeObservation,
    FrameStackObservation(n_stack=4),
    ClipReward,
]
```

**Training telemetry:**

```python
wrappers=[
    RecordEpisodeStatistics,  # info["episode"]
]
```

**Evaluation with video:**

```python
wrappers=[
    RecordVideo,  # MP4 per episode, not JIT-compatible
]
```

## Common Pitfalls

Here are some common "gotchas" to be mindful of:

- **Applying `RecordVideo` inside a JIT boundary**. Don't do it. It writes Python-side files and should only be used for evaluation purposes, outside `jax.jit`/`jax.vmap`.
- **Wrong input shape for `GrayscaleObservation`**. This wrapper expects `uint8[H, W, 3]`. If your environment outputs `floats` or grayscale already, you get a shape/dtype error at trace time.
- **Ordering `NormalizeObservation` before `GrayscaleObservation` / `ResizeObservation`**. The `NormalizeObservation` wrapper turns `uint8[0, 255]` observations into `float32[0, 1]`. The shape transforms expect `uint8`. Perform shape transforms first, then normalize them.

## Recap

Excellent job! To recap:

- We have two types of wrappers: **pass-through** (stateless) and **stateful** (introduces an outer state type wrapping the inner state).
- Envrax comes with 8 pass-through wrappers: `JitWrapper`, `GrayscaleObservation`, `ResizeObservation`, `NormalizeObservation`, `ClipReward`, `EpisodeDiscount`, `ExpandDims`, `RecordVideo`
- And 2 stateful wrappers: `FrameStackObservation`, `RecordEpisodeStatistics`
- We apply wrappers via `make(wrappers=[...])` (innermost-first) or manual composition
- Parameterised wrappers can be passed to `make()` methods without the `env` parameter. No `functools.partial` required!
- `env.unwrapped` provides the innermost (base) `JaxEnv`
- Always do shape transforms (`Grayscale`, `Resize`) on `uint8` observations before `NormalizeObservation` casts them to `float32`

## Next Steps

For our last tutorial, we'll look at how to use the `render()` method so that you can watch your agents in their environments. See you there! :wave:

<div class="grid cards" markdown>

-   :material-image-outline:{ .lg .middle } __Rendering__

    ---

    Learn how to use the `render()` method.

    [:octicons-arrow-right-24: Continue to Tutorial 9](rendering.md)

</div>
