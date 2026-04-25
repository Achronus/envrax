import chex
import jax
import jax.numpy as jnp

from envrax.env import EnvConfig, EnvState, JaxEnv
from envrax.multi_vec_env import MultiVecEnv
from envrax.spaces import Box, Discrete
from envrax.vec_env import VecEnv

# ---------------------------------------------------------------------------
# Two minimal envs with different obs shapes
# ---------------------------------------------------------------------------


@chex.dataclass
class _VecState(EnvState):
    pass


class _VecEnv(JaxEnv[Box, Discrete, _VecState, EnvConfig]):
    """float32[4] observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=(4,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        return obs, _VecState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))

    def step(self, state, action):
        rng, _ = jax.random.split(state.rng)
        s = state.__replace__(rng=rng, step=state.step + 1)
        return jnp.zeros((4,), jnp.float32), s, jnp.float32(1.0), jnp.bool_(False), {}


@chex.dataclass
class _PixState(EnvState):
    pass


class _PixEnv(JaxEnv[Box, Discrete, _PixState, EnvConfig]):
    """uint8[4, 4, 3] observations."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=255, shape=(4, 4, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=3)

    def reset(self, rng):
        obs = jnp.full((4, 4, 3), 128, dtype=jnp.uint8)
        return obs, _PixState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))

    def step(self, state, action):
        rng, _ = jax.random.split(state.rng)
        s = state.__replace__(rng=rng, step=state.step + 1)
        return jnp.full((4, 4, 3), 64, jnp.uint8), s, jnp.float32(1.0), jnp.bool_(False), {}


_RNG = jax.random.key(0)
_CONFIG = EnvConfig(max_steps=10)
_N_ENVS = 4


class TestMultiVecEnv:
    def test_num_envs(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        assert multi.num_envs == 2

    def test_total_envs(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), 4),
            VecEnv(_PixEnv(config=_CONFIG), 8),
        ])
        assert multi.total_envs == 12

    def test_vec_envs_property(self):
        vecs = [
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ]
        multi = MultiVecEnv(vecs)
        assert multi.vec_envs is vecs

    def test_reset_returns_lists(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        obs_list, state_list = multi.reset(_RNG)
        assert isinstance(obs_list, list)
        assert len(obs_list) == 2
        assert len(state_list) == 2

    def test_reset_obs_have_batch_dim(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        obs_list, _ = multi.reset(_RNG)
        assert obs_list[0].shape == (_N_ENVS, 4)
        assert obs_list[1].shape == (_N_ENVS, 4, 4, 3)

    def test_step_returns_correct_structure(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        _, states = multi.reset(_RNG)
        actions = [
            jnp.zeros(_N_ENVS, dtype=jnp.int32),
            jnp.zeros(_N_ENVS, dtype=jnp.int32),
        ]
        obs, new_states, rewards, dones, infos = multi.step(states, actions)
        assert len(obs) == 2
        assert len(rewards) == 2
        assert rewards[0].shape == (_N_ENVS,)
        assert dones[1].shape == (_N_ENVS,)

    def test_reset_at(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        obs, state = multi.reset_at(1, _RNG)
        assert obs.shape == (_N_ENVS, 4, 4, 3)

    def test_step_at(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
        ])
        _, states = multi.reset(_RNG)
        actions = jnp.zeros(_N_ENVS, dtype=jnp.int32)
        obs, _, rewards, _, _ = multi.step_at(0, states[0], actions)
        assert obs.shape == (_N_ENVS, 4)
        assert rewards.shape == (_N_ENVS,)

    def test_class_groups(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        groups = multi.class_groups
        assert "_VecEnv" in groups
        assert "_PixEnv" in groups
        assert groups["_VecEnv"] == [0, 1]
        assert groups["_PixEnv"] == [2]

    def test_single_spaces(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        assert multi.single_observation_spaces[0].shape == (4,)
        assert multi.single_observation_spaces[1].shape == (4, 4, 3)

    def test_empty_raises(self):
        import pytest

        with pytest.raises(ValueError, match="at least one"):
            MultiVecEnv([])

    def test_compile(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        multi.compile(progress=False)
        # Should work after compile
        obs_list, _ = multi.reset(_RNG)
        assert obs_list[0].shape == (_N_ENVS, 4)
        assert obs_list[1].shape == (_N_ENVS, 4, 4, 3)

    def test_repr(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), 4),
            VecEnv(_PixEnv(config=_CONFIG), 8),
        ])
        r = repr(multi)
        assert "MultiVecEnv" in r
        assert "total=12" in r
