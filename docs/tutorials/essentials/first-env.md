# Your First Environment

???+ api "API Docs"

    [`envrax.env.JaxEnv`](../../api/env/base.md#envrax.env.JaxEnv)

Welcome back! So far, you've developed your understanding of the three foundational pieces for building Envrax environments:

- [State](state.md) — the immutable snapshot of your environment
- [Spaces](spaces.md) — the contracts describing observations and actions
- [Configuration](configuration.md) — the static settings that drive its dynamics

Now, it's time to wire them together into a working environment!

We'll build a tiny 2D *ball* world where a ball starts at a random location and an agent takes one of four discrete actions per step: `[left, right, up, down]`. By the end you'll have a runnable `JaxEnv` and understand the `reset` / `step` contract that every Envrax environment follows.

Without further ado, let's get to it! :rocket:

## Fundamental Components

From our first tutorial ([State](state.md)) we already created our `BallState`, here's a refresher:

```python
@chex.dataclass
class BallState(EnvState):
    ball_x: chex.Array
    ball_y: chex.Array
```

We'll also reuse the `BallConfig` from [Configuration](configuration.md):

```python
@chex.dataclass
class BallConfig(EnvConfig):
    friction: float = 0.98
    reward_scale: float = 1.0
```

What we didn't discuss was the types of `Spaces` we were going to use. Recall that we need two: an `observation` space and an `action` space.

Based on our initial brief, the action space is easy - `Discrete(n=4)` to cover our four movement options.

However, the observation space is a little trickier. To help with this, let's consider the following:

1. **What does the agent see?** - we want the agent to be able to see how the ball moves towards a target, so we'll need it to monitor its `(x, y)` position.
2. **What format are the positions in?** - could we set them up as a `Discrete` space or a `Box`? Here we use `float` values so `Box` would come naturally.
3. **What value range do we need?** - this defines the "world" the ball lives in and is purely a design choice. We could use absolute coordinates (e.g., pixel positions with a range of `[0, 800]`), but this is unnecessary complexity. Instead, we can use a normalized range between `0.0` and `1.0`. This is a natural fit for Neural Networks too!

That gives us `Box(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)` - a continuous 2-vector bounded observation space with values between `0` and `1`, matching the `jnp.float32` dtype we used on `BallState`.

Perfect! Now we have everything we need. Let's build our `BallEnv`!

## Building the Environment

??? example "Full Code"

    If needed, here's the full code used throughout this tutorial. Drop it into a file called `ball_env.py` and run it:

    ```python
    import chex
    import jax
    import jax.numpy as jnp

    from envrax import JaxEnv, EnvState, EnvConfig
    from envrax.spaces import Box, Discrete


    @chex.dataclass
    class BallState(EnvState):
        ball_x: chex.Array
        ball_y: chex.Array


    @chex.dataclass
    class BallConfig(EnvConfig):
        friction: float = 0.98
        reward_scale: float = 1.0


    class BallEnv(JaxEnv[Box, Discrete, BallState, BallConfig]):
        @property
        def observation_space(self) -> Box:
            return Box(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)

        @property
        def action_space(self) -> Discrete:
            return Discrete(n=4)

        def reset(self, rng: chex.PRNGKey):
            rng, init_rng = jax.random.split(rng)
            rng_x, rng_y = jax.random.split(init_rng)
            state = BallState(
                rng=rng,
                step=jnp.int32(0),
                done=jnp.bool_(False),
                ball_x=jax.random.uniform(rng_x),
                ball_y=jax.random.uniform(rng_y),
            )
            obs = jnp.array([state.ball_x, state.ball_y])
            return obs, state

        def step(self, state: BallState, action: chex.Array):
            rng, _ = jax.random.split(state.rng)

            # Use action to get new obs
            # action: 0=left, 1=right, 2=up, 3=down
            dx = jnp.array([-0.01, 0.01, 0.0, 0.0])[action] * self.config.friction
            dy = jnp.array([0.0, 0.0, -0.01, 0.01])[action] * self.config.friction

            # Get bounds
            low, high = self.observation_space.low, self.observation_space.high

            # Increment obs
            new_x = jnp.clip(state.ball_x + dx, low, high)
            new_y = jnp.clip(state.ball_y + dy, low, high)

            # Update new state
            new_state = state.replace(
                rng=rng,
                step=state.step + 1,
                ball_x=new_x,
                ball_y=new_y,
            )

            # Set new obs
            obs = jnp.array([new_state.ball_x, new_state.ball_y])

            # Compute reward, done, and info
            reward = jnp.float32(1.0) * self.config.reward_scale
            done = new_state.step >= self.config.max_steps
            info = {"current_step": new_state.step}

            return obs, new_state.replace(done=done), reward, done, info


    if __name__ == "__main__":
        # Init the environment
        env = BallEnv()

        # Set it's initial state
        obs, state = env.reset(jax.random.key(42))
        print("obs:", obs)             # shape (2,) — starting position
        print("step:", state.step)     # 0
        print("done:", state.done)     # False

        # Run a step!
        obs, state, reward, done, info = env.step(state, action=jnp.int32(0))
        print("reward:", reward)       # 1.0
        print("step:", state.step)     # 1
    ```

    This code should work "as is".

We can build an Envrax environment in three easy steps:

1. Choosing a class name and assigning the generic types
2. Defining the environments spaces
3. Implementing the methods - `reset` and `step`

### Step 1: Declaring our Class

??? abstract "`JaxEnv` Base Class"

    Curious what's under the hood? Here's `JaxEnv` stripped to its essentials:

    ??? example "`JaxEnv` Code"

        ```python
        from abc import ABC, abstractmethod
        from typing import Generic, Tuple, TypeVar

        from envrax.spaces import Space


        ObsSpaceT = TypeVar("ObsSpaceT", bound=Space)
        ActSpaceT = TypeVar("ActSpaceT", bound=Space)
        StateT = TypeVar("StateT", bound="EnvState")
        ConfigT = TypeVar("ConfigT", bound="EnvConfig")


        class JaxEnv(ABC, Generic[ObsSpaceT, ActSpaceT, StateT, ConfigT]):
            @property
            @abstractmethod
            def observation_space(self) -> ObsSpaceT: ...

            @property
            @abstractmethod
            def action_space(self) -> ActSpaceT: ...

            @abstractmethod
            def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, StateT]: ...

            @abstractmethod
            def step(
                self, state: StateT, action: chex.Array,
            ) -> Tuple[chex.Array, StateT, chex.Array, chex.Array, Dict[str, Any]]: ...
        ```

    Two things worth mentioning:

    - **`ABC`** — marks the class as *abstract*, forcing subclasses to implement every method marked with `@abstractmethod` before they can be instantiated.
    - **`Generic[ObsSpaceT, ActSpaceT, StateT, ConfigT]`** — declares four type parameters, each `bound` to its base type (`Space`, `EnvState`, or `EnvConfig`). So, when you write `JaxEnv[Box, Discrete, BallState, BallConfig]`, you're *pinning* those TypeVars to concrete types for this subclass. This allows your IDE to know which type is being used and can perform autocompletion correctly without hacky overrides or `# type: ignore`.

Every Envrax environment **must** subclass `JaxEnv` and pin four data types for IDE support. These are (in order): the observation space, action space, the environment state, and the environment config.

In our case, we have `Box`, `Discrete` and our custom `BallState`:

```python
from envrax import JaxEnv
from envrax.spaces import Box, Discrete

class BallEnv(JaxEnv[Box, Discrete, BallState, BallConfig]): # (1)
    ...
```

1. Format: `[observation_space, action_space, EnvState]`

### Step 2: Defining our Spaces

Next, we declare the `observation_space` and `action_space` as properties on the class:

```python
class BallEnv(JaxEnv[Box, Discrete, BallState, BallConfig]):
    @property
    def observation_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=4)
```

### Step 3: Implement our Methods

Before writing any code, let's first consider what the `reset` and `step` methods actually do:

- `reset` - takes a `jax.random.key()` and outputs an **initial observation** and an **initial `EnvState`**.
- `step` - takes the current `EnvState` and an agents `action` and iterates through the environment to transition to a **new observation**, produces a **new `EnvState`**, provides a **reward**, a **termination result** defining whether the environment has ended, and additional **metadata**.

#### Reset Method

`reset` is the easier of the two, so we'll start there. Looking at our description, we can unpack it into three key steps:

1. Handling the PRNG key
2. Creating the initial state
3. Creating the initial observation

For the PRNG key, we split it once into two keys - the first for the `BallState` and the second for splitting again to create the balls *random starting position* (the `x` and `y` positions).

Here's what the first part looks like:

```python
def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, BallState]:
    rng, init_rng = jax.random.split(rng) # (1)
    rng_x, rng_y = jax.random.split(init_rng) # (2)
    ...
```

1. The `BallState` key and the `position` splitting key
2. The `x` and `y` RNG keys

Now, we can create the initial state with starting values using the random keys:

```python
    ...
    state = BallState(
        rng=rng,
        step=jnp.int32(0),
        done=jnp.bool_(False),
        ball_x=jax.random.uniform(rng_x),
        ball_y=jax.random.uniform(rng_y),
    )
    ...
```

Since we are using `jnp.float32` values for the balls `(x, y)` position, we sample from a [uniform [:material-arrow-right-bottom:]](https://docs.jax.dev/en/latest/_autosummary/jax.random.uniform.html) distribution to get a random starting state that's different with every key.

Finally, we can create the initial observation using the initial positions and return the required values:

```python
    ...
    obs = jnp.array([state.ball_x, state.ball_y])
    return obs, state
```

Great! That's the `reset` method done! :smile:

Here's what it looks like in full:

```python
def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, BallState]:
    rng, init_rng = jax.random.split(rng)
    rng_x, rng_y = jax.random.split(init_rng)

    state = BallState(
        rng=rng,
        step=jnp.int32(0),
        done=jnp.bool_(False),
        ball_x=jax.random.uniform(rng_x),
        ball_y=jax.random.uniform(rng_y),
    )
    obs = jnp.array([state.ball_x, state.ball_y])
    return obs, state
```

#### Step Method

Now, the `step` method. Recall that:

> `step` - takes the current `EnvState` and an agents `action` and iterates through the environment to transition to a **new observation**, produces a **new `EnvState`**, provides a **reward**, a **termination result** defining whether the environment has ended, and additional **metadata**.

Yikes! :face_with_peeking_eye: There's a lot to unpack there so let's think about this carefully. We need to:

1. Manage the PRNG randomness to get a new observation and state (required for JAX)
2. Create a new `EnvState`
3. Take an action through the environment to create a new observation
4. Get a reward signal
5. Check if the environment is done
6. Get the metadata for the environment step
7. Return the required values

That's a lot! Let's take it one step at a time, starting with the PRNG management. For this, we want to extract the `rng` key from the provided `state` and split it for reuse during for the next timestep.

We can do this in one line using our handy-dandy `jax.random.split()` approach:

```python
def step(self, state: BallState, action: chex.Array):
    rng, _ = jax.random.split(state.rng)
    ...
```

Easy enough! Next, let's create the new `EnvState` and observation.

Here, we'll create static lookup tables for `x` and `y` and extract the corresponding value based on action value as our index. For example, if `action=0`, `x=-0.01`, `y=0.0`.

Then, we'll use `jnp.clip` to increment our ball state while keeping its values in the bounds of the observation space:

```python
    ...
    # Use action to get new obs
    # action: 0=left, 1=right, 2=up, 3=down
    dx = jnp.array([-0.01, 0.01, 0.0, 0.0])[action] * self.config.friction
    dy = jnp.array([0.0, 0.0, -0.01, 0.01])[action] * self.config.friction

    # Get bounds
    low, high = self.observation_space.low, self.observation_space.high

    # Increment obs
    new_x = jnp.clip(state.ball_x + dx, low, high)
    new_y = jnp.clip(state.ball_y + dy, low, high)
    ...
```

Notice how we've used the `self.config.friction` config field here (`BallConfig.friction`). To give that real ball feel, every per-step displacement is scaled by friction. If we reduce it to `0.5`, the ball will move more sluggishly, but if we bump it up to `1.0`, it moves at full speed.

If we wanted, we could separate this out into a separate `_act()` method on the environment class to keep our `step()` method easy to read. We won't do that here for this simple tutorial, but something to think about when building more complex ones! :wink:

Now, we use the `.replace()` method to update the `EnvState` and create the observation just like the initial one but with our `new_state` instead:

```python
    ...
    # Update new state
    new_state = state.replace(
        rng=rng,
        step=state.step + 1,
        ball_x=new_x,
        ball_y=new_y,
    )

    # Set new obs
    obs = jnp.array([new_state.ball_x, new_state.ball_y])
    ...
```

Notice how we incremented our `step` here so that we can track things accordingly. Okay, 3/7 down! Next, the reward signal, our done flag and the metadata.

For this example, we'll give our agent a flat `1.0` per step, scaled by `self.config.reward_scale` from our `BallConfig`. Reward function creation and reward shaping is a beast of its own that is out of the scope of this tutorial series. Google DeepMind provide a great post about [Specification Gaming [:material-arrow-right-bottom:]](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) that highlights some of the challenges when building reward functions. Highly recommend considering it when building your own!

For our termination flag, we'll simply check to see if the current step matches the `config.max_steps` for our `BallConfig` (inherited from `EnvConfig`).

For our metadata we'll just return a Python `Dict` with the current step count.

Here's what all of that looks like:

```python
    ...
    reward = jnp.float32(1.0) * self.config.reward_scale
    done = new_state.step >= self.config.max_steps
    info = {"current_step": new_state.step}
    ...
```

??? info "Customization"

    These three values (`reward`, `done`, `info`) can be far more complicated and customized depending on your environments complexity. 
    
    It's not uncommon to extract these into their own full-blown helper methods such as `_reward()`, `_done()`, and `_info`, just like an `_act()` function. In fact, it's a good practice to do so!
    
    Remember to check out Envrax's built-in [Wrappers](wrappers.md) to find some existing customization options too!

Lastly, all we need to do is return the values. Here's the complete method with the return statement included:

```python
def step(self, state: BallState, action: chex.Array):
    rng, _ = jax.random.split(state.rng)

    # Use action to get new obs
    # action: 0=left, 1=right, 2=up, 3=down
    dx = jnp.array([-0.01, 0.01, 0.0, 0.0])[action] * self.config.friction
    dy = jnp.array([0.0, 0.0, -0.01, 0.01])[action] * self.config.friction

    # Get bounds
    low, high = self.observation_space.low, self.observation_space.high

    # Increment obs
    new_x = jnp.clip(state.ball_x + dx, low, high)
    new_y = jnp.clip(state.ball_y + dy, low, high)

    # Update new state
    new_state = state.replace(
        rng=rng,
        step=state.step + 1,
        ball_x=new_x,
        ball_y=new_y,
    )

    # Set new obs
    obs = jnp.array([new_state.ball_x, new_state.ball_y])
    
    # Compute reward, done, and info
    reward = jnp.float32(1.0) * self.config.reward_scale
    done = new_state.step >= self.config.max_steps
    info = {"current_step": new_state.step}

    return obs, new_state.replace(done=done), reward, done, info
```

Notice how we return an updated copy of our `new_state` with the updated `done` flag here to simplify our method a little more.

## Running It

Nice work so far! Now let's try running this bad boy. :muscle:

We can do that in 3 lines of code + a few `print()` statements for verification:

```python
import jax
import jax.numpy as jnp

from my_project.env import BallEnv

# Init the environment
env = BallEnv()

# Set it's initial state
obs, state = env.reset(jax.random.key(42))
print("obs:", obs)             # shape (2,) — starting position
print("step:", state.step)     # 0
print("done:", state.done)     # False

# Run a step!
obs, state, reward, done, info = env.step(state, action=jnp.int32(0))
print("reward:", reward)       # 1.0
print("step:", state.step)     # 1
```

That's it! The full `reset → step` loop! :star_struck:

## Using Wrappers

Envrax ships with a set of *wrappers* that transform observations, rewards, or termination flags without touching your env's code. They're applied like onion layers - each takes an inner env and returns a new one with the same `reset`/`step` interface but with added functionality (where appropriate):

```python
from envrax.wrappers import ClipReward, NormalizeObservation

env = BallEnv()
env = NormalizeObservation(env)    # observations → float32 in [0, 1]
env = ClipReward(env)              # reward → sign(reward)
```

For production setups, the [`make()`](make.md) factory method is useful for doing this automatically:

```python
import envrax

env = envrax.make(
    "BallEnv-v0",
    wrappers=[NormalizeObservation, ClipReward],
)
```

We'll dive into specific wrappers in [Available Wrappers](wrappers.md) and walk through every factory method in the [Make Methods](make.md) tutorial. For now, just know they exist! :wink:

## Recap

Excellent job! You've just built your first `JaxEnv` environment! :partying_face:

Here's a quick recap of what we've covered:

- [x] Declared a `BallEnv` class subclassing `JaxEnv[Box, Discrete, BallState, BallConfig]`
- [x] Defined the `observation_space` and `action_space` as properties on the class
- [x] Implemented the `reset` and `step` methods to drive the environment's transitions
- [x] Tested it by running the `reset → step` loop

Next, we'll explore the `EnvConfig` and how to customize it.

## Next Steps

<div class="grid cards" markdown>

-   :material-vector-arrange-above:{ .lg .middle } __Vectorising with `VecEnv`__

    ---

    Learn how to run `N` parallel copies of your environments.

    [:octicons-arrow-right-24: Continue to Tutorial 5](vectorising.md)

</div>
