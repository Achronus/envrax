# Getting Started

To get started, setup a Python 3.13+ environment and install the package.

## Project Setup

If you don't already have a Python project, spin one up with your tool of choice:

=== "uv"
    ```bash title=""
    uv init --python 3.13 my-project
    cd my-project
    ```

=== "pip (Linux/macOS)"
    ```bash title=""
    mkdir my-project && cd my-project
    python3.13 -m venv .venv
    source .venv/bin/activate
    ```

=== "pip (Windows)"
    ```bash title=""
    mkdir my-project && cd my-project
    py -3.13 -m venv .venv
    .venv\Scripts\activate
    ```

=== "poetry"
    ```bash title=""
    poetry new --python ">=3.13" my-project
    cd my-project
    ```

## Install Package

Then, install the package:

=== "uv"
    ```bash title=""
    uv add envrax
    ```

=== "pip"
    ```bash title=""
    pip install envrax
    ```

=== "poetry"
    ```bash title=""
    poetry add envrax
    ```

If you're new, or want a refresher, head on over to the [tutorials](../tutorials/index.md) or try out the example below!

## Example Usage

A simple ball environment example:

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
    key = jax.random.key(42)
    obs, state = env.reset(key)

    # Iterate through 1000 timesteps
    for _ in range(1000):
        action = env.action_space.sample(key)
        obs, state, reward, done, info = env.step(state, action)

        # If episode has ended, reset to start a new one
        if done:
            new_key, key = jax.random.split(key)
            obs, info = env.reset(new_key)
```

This code should work "as is".

### Make Parallel Copies of It

```python
import jax
import jax.numpy as jnp
from envrax import VecEnv, EnvConfig

vec_env = VecEnv(BallEnv(config=EnvConfig(max_steps=1000)), num_envs=512)
obs, states = vec_env.reset(jax.random.key(42))   # obs: float32[512, 2]

actions = jnp.zeros(512, dtype=jnp.int32)
obs, states, rewards, dones, infos = vec_env.step(states, actions)
# rewards: float32[512]
# dones:   bool[512]
```

This code should work "as is".

## Next Steps

<div class="grid cards" markdown>

-   :material-creation-outline:{ .lg .middle } __Tutorials__

    ---

    Learn how to use Envrax, your way!

    [:octicons-arrow-right-24: Start learning](../tutorials/index.md)

-   :fontawesome-solid-paper-plane:{ .lg .middle } __API__

    ---

    Explore the code making Envrax possible.

    [:octicons-arrow-right-24: Explore the API](../api/index.md)

</div>
