# Rendering

To truly understand something we need to *see* it in action. Let's say your agent isn't learning. Is that because the agent is broken or the environment isn't working as intended?

The only way to know for sure is to *observe* how your agent operates inside your environment. This is where rendering comes in.

Rendering turns an environment state (`observation`) into a snapshot (picture) that allows you to *observe* what the agent is doing and how it's interacting with the environment at a point in time.

It's completely optional but an invaluable tool to add to your arsenal for effectively debugging your agents and environments, logging, and recording videos.

Throughout this tutorial, we'll discuss how we can apply this to our Envrax environments using a `render()` method. Let's get started! :muscle:

## `render()`

???+ api "API Docs"

    [`envrax.env.JaxEnv.render`](../../api/env/base.md#envrax.env.JaxEnv.render)

By default, `JaxEnv` provides a blank `render(state)` method that **must** be implemented on your own environments before you can use it.

It's completely optional and isn't a mandatory requirement for environments but is highly valuable to implement. By default, environments that don't override the `render()` method raise a `NotImplementedError`.

For example, let's expand the `BallEnv` we created earlier and give it a simple `render()` method:

```python
import numpy as np

class BallEnv(JaxEnv[Box, Discrete, BallState, EnvConfig]):
    def render(self, state: BallState) -> np.ndarray:
        H, W = 210, 160
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Read scalars from state once
        x = float(state.ball_x)
        y = float(state.ball_y)

        # Draw a 5×5 white ball at the (x, y) position
        cx, cy = int(x * W), int(y * H)
        frame[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3] = 255

        return frame
```

???+ warning "Use NumPy arrays instead of JAX"
    For compatibility and usability with other packages such as [Pillow [:material-arrow-right-bottom:]](https://pillow.readthedocs.io/), [matplotlib [:material-arrow-right-bottom:]](https://matplotlib.org/), and [Jupyter Notebooks [:material-arrow-right-bottom:]](https://jupyter.org/), the `render()` method should return a **NumPy** (not JAX) array of shape `(H, W, 3)` and datatype of `uint8`.

We can then run it to produce a single frame:

```python
import jax

env = BallEnv()
_, state = env.reset(jax.random.key(0))
frame = env.render(state)
print(frame.shape, frame.dtype)   # (210, 160, 3) uint8
```

And save it to PNG or display it with whatever image library you prefer:

```python
import imageio
imageio.imwrite("ball.png", frame)
```

## Rendering from a `VecEnv`

`VecEnv` supports rendering by default with its own `render()` method. It uses the underlying environment's `render()` method and assumes that it has already been implemented. If not, it simply falls back to the base case and raises a `NotImplementedError`.

To use it, you need to provide a `state` and specify the `index` of the environment that you want to extract from the batched state PyTree:

```python hl_lines="4"
vec_env = VecEnv(BallEnv(), num_envs=64)
obs, states = vec_env.reset(jax.random.key(0))

frame = vec_env.render(states, index=0)  # Renders env 0 from the batch
```

No need to craft this behaviour by hand! :wink:

## Rendering Through Wrappers

Wrappers forward `render()` to their inner env by default, so you'll never lose rendering capabilities:

```python
env = ClipReward(ResizeObservation(BallEnv(), h=84, w=84))
frame = env.render(state)  # Reaches through to BallEnv.render
```

This applies even to stateful wrappers like `FrameStackObservation`, because the wrapper state keeps a forwarded copy of the inner state that gets passed down the chain.

## Design Tips

Here are some design tips to consider when building your `render()` methods:

- **Keep `render` fast but not JIT-compiled.** `render` is called at human timescales (once per frame for video, once per log step). Don't JIT it! Keep it in NumPy so that any downstream drawing libraries (Pillow, OpenCV, matplotlib) can use it without JAX restrictions.
- **Avoid dynamic shapes.** Always return the same `(H, W, 3)` shape. If you need a HUD or overlay, draw it into the fixed canvas rather than concatenating arrays of varying sizes.
- **Pull scalars once.** Calling `float(state.field)` forces a device → host transfer. To maximise performance, grab all the values you need at the top of `render` and then create the renderable frame.
- **Cache expensive static assets.** If your render draws a fixed background (e.g. a maze layout or a level tileset), cache it on `self` in the `__init__` method and copy it per frame.

## Rendering as MP4

Once `render` is implemented, you can wrap your environment in a [`RecordVideo`](wrappers.md#recordvideo) wrapper.

This automatically captures episode rollouts to MP4 for you:

```python
from envrax.wrappers import RecordVideo

env = RecordVideo(BallEnv(), output_dir="runs/recordings/")
# Every episode ends → MP4 saved to runs/recordings/episode_NNNN.mp4
```

See the [`RecordVideo`](wrappers.md#recordvideo) section of the wrappers tutorial for the full set of trigger options.

## Common Pitfalls

Be wary of the following "gotchas":

- **Returning `jnp.ndarray`**. Image libraries expect NumPy, and as part of the API standard, you should always return an `np.ndarray` from the `render()` method.
- **Wrong dtype**. Rendered frames **must** be `uint8`, not `float32`. If you normalised them, multiply them by `255` and cast them to `uint8` before returning the frames.
- **Trying to render from inside `jax.jit`**. `render` reads JAX scalars Python-side, which triggers a `ConcretizationTypeError` under tracing. Always use `render()` *outside* the JIT boundary.
- **`NotImplementedError` when wrapped**. Oops! You've forgotten to build a custom `render()` method. Simply override the inherited `render()` method on your environment class.

## Recap

And there we have it! That's how to use visual rendering in your environments. To recap:

- `render(state)` **must** return an `np.ndarray` uint8 of shape `(H, W, 3)`
- `render()` runs Python-side and should not be used inside JIT so that it can support any CPU drawing library
- Wrappers forward `render()` by default
- `VecEnv.render(state, index=[int])` picks one env from a batch
- Implementing a `render()` method enables video recording, episode logging, and visual debugging

## Essentials Series: Summary

Excellent work! :clap:

It's official, that was the last of the **Essentials** tutorials. You now know all the core details needed to use Envrax to your heart's content!

Here's a final recap of what we've covered in this series:

- **Environment State**: You learned how to model environment data as immutable, JAX-traceable PyTrees by extending `EnvState`, and how to thread a PRNG key through the episode so randomness stays deterministic across every `reset` and `step` call.
- **Spaces**: You explored how to describe what your agent sees and how it acts using `Box`, `Discrete`, and `MultiDiscrete` as pure metadata contracts that let downstream code shape policy networks and catch shape mismatches before they hit your training loop.
- **Environment Configuration**: You learned the static vs. traceable split — `EnvConfig` for one-time settings declared at construction, `EnvState` for everything that changes through an episode. Keeping them separate is what stops JIT from silently re-compiling on you.
- **Your First Environment**: You built an end-to-end `JaxEnv` from scratch by pinning the four type generics, defining your spaces, and implementing the `reset`/`step` contract that drives every Envrax environment.
- **Vectorising with `VecEnv`**: You learned how to spin up hundreds of parallel environments in a single line via `jax.vmap`, with auto-reset on `done=True` handled inside the vmapped body so you never have to branch on episode boundaries yourself.
- **Multiple Environments**: You've seen how to compose `M` heterogeneous environments into a single managed fleet using `MultiEnv` / `MultiVecEnv`, making multi-task training, meta-learning, and heterogeneous evaluation suites that tiny bit easier.
- **Environment Registry**: You learned how to expose environments under canonical names through `register()` / `register_suite()` so your training, evaluation, and analysis scripts all share one source of truth instead of importing classes directly.
- **Make Methods**: You learned how to use Envrax's single-line builders — `make()`, `make_vec()`, `make_multi()`, and `make_multi_vec()` — which handle wrappers, JIT compilation, and resolved configs automatically for you.
- **Available Wrappers**: You learned the difference between pass-through and stateful wrappers, when to reach for each, and how to chain them together to build classic preprocessing pipelines like the Atari image stack.
- **Rendering**: And with this tutorial, you learned how to turn any environment state into a viewable RGB frame with `render(state)` and feed it straight into video recording, training-time visualisations, or quick debugging sessions.

## Next Steps

So, where to next? :sparkles:

You've now got enough knowledge to use the full extent of the Envrax package. At this point, you should really start building!

But if you need some extra help on unique topics we highly recommend checking out our **Advanced** tutorials. Alternatively, you can also check out our API reference to look up specific classes and methods.

Happy building! :rocket:

<div class="grid cards" markdown>

-   :material-tools:{ .lg .middle } __Advanced Recipes__

    ---

    Task-focused walkthroughs for more advanced topics and customization options.

    [:octicons-arrow-right-24: Browse advanced recipes](../index.md#advanced)

-   :fontawesome-solid-paper-plane:{ .lg .middle } __API Reference__

    ---

    Look up specific classes, methods, and parameter signatures that Envrax uses.

    [:octicons-arrow-right-24: Browse the API](../../api/index.md)

</div>
