# Getting Started

To get started, setup a Python 3.13+ environment and install the package through [pip [:material-arrow-right-bottom:]](https://pypi.org/project/envrax/):

```bash
pip install envrax
```

Then, head on over to the [tutorials](../tutorials/index.md) or try out the example below!

## Example Usage

A simple ball environment example:

```python
import chex
import jax
import jax.numpy as jnp

from envrax import JaxEnv, EnvState
from envrax.spaces import Box, Discrete


@chex.dataclass
class BallState(EnvState):
    ball_x: jnp.float32
    ball_y: jnp.float32


class BallEnv(JaxEnv[Box, Discrete, BallState]):
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
        new_state = state.replace(rng=rng, step=state.step + 1)
        obs = jnp.array([new_state.ball_x, new_state.ball_y])
        reward = jnp.float32(1.0)
        done = new_state.step >= self.config.max_steps
        return obs, new_state.replace(done=done), reward, done, {}
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
