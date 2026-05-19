from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envrax.batched_env import BatchedEnv
from envrax.spaces import Box, Discrete, Space


class _MinimalBatched(BatchedEnv):
    """Smallest concrete BatchedEnv that satisfies every abstract method."""

    n_slots = 3

    @property
    def single_observation_space(self) -> Space:
        return Box(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)

    @property
    def single_action_space(self) -> Space:
        return Discrete(n=2)

    def reset(self, rng: chex.PRNGKey) -> Tuple[jax.Array, Any]:
        obs = jnp.zeros((self.n_slots, 2), dtype=jnp.float32)
        state = {"step": jnp.zeros(self.n_slots, dtype=jnp.int32)}
        return obs, state

    def step(
        self, state: Any, actions: jax.Array
    ) -> Tuple[jax.Array, Any, jax.Array, jax.Array, Dict[str, Any]]:
        new_state = {"step": state["step"] + 1}
        return (
            jnp.zeros((self.n_slots, 2), dtype=jnp.float32),
            new_state,
            jnp.zeros((self.n_slots,), dtype=jnp.float32),
            jnp.zeros((self.n_slots,), dtype=jnp.bool_),
            {},
        )

    def slot_state(self, state: Any, slot_idx: int) -> Any:
        return {"step": state["step"][slot_idx]}

    def render_slot(self, state: Any, slot_idx: int) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def compile(self, cache_dir=None) -> None:
        pass


class TestBatchedEnvContract:
    def test_minimal_implementation_instantiates(self):
        env = _MinimalBatched()
        assert env.n_slots == 3

    def test_missing_abstract_method_raises_typeerror(self):
        class _Partial(BatchedEnv):
            n_slots = 1

            @property
            def single_observation_space(self) -> Space:
                return Box(low=0, high=1, shape=(1,), dtype=jnp.float32)

        with pytest.raises(TypeError, match="abstract"):
            _Partial()

    def test_reset_returns_obs_and_state(self):
        env = _MinimalBatched()
        obs, state = env.reset(jax.random.key(0))
        assert obs.shape == (3, 2)
        assert state["step"].shape == (3,)

    def test_step_shapes(self):
        env = _MinimalBatched()
        _, state = env.reset(jax.random.key(0))
        actions = jnp.zeros((3,), dtype=jnp.int32)
        obs, new_state, reward, done, info = env.step(state, actions)
        assert obs.shape == (3, 2)
        assert reward.shape == (3,)
        assert done.shape == (3,)
        assert new_state["step"].tolist() == [1, 1, 1]
        assert isinstance(info, dict)

    def test_slot_state_removes_leading_dim(self):
        env = _MinimalBatched()
        _, state = env.reset(jax.random.key(0))
        single = env.slot_state(state, 1)
        assert single["step"].shape == ()

    def test_render_slot_returns_rgb_frame(self):
        env = _MinimalBatched()
        _, state = env.reset(jax.random.key(0))
        frame = env.render_slot(state, 0)
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3


class TestBatchedEnvName:
    def test_default_name_is_subclass_name(self):
        assert _MinimalBatched().name == "_MinimalBatched"


class TestVecEnvSatisfiesBatchedEnv:
    """VecEnv is the canonical BatchedEnv impl — confirm protocol conformance."""

    def test_vec_env_is_batched_env_subclass(self):
        from envrax.vec_env import VecEnv
        assert issubclass(VecEnv, BatchedEnv)

    def test_vec_env_instance_is_batched_env(self):
        from envrax.env import EnvConfig
        from envrax.vec_env import VecEnv
        from tests.test_multi_vec_env import _VecEnv

        v = VecEnv(_VecEnv(config=EnvConfig(max_steps=5)), 4)
        assert isinstance(v, BatchedEnv)
        assert v.n_slots == 4

    def test_vec_env_name_overrides_default_with_inner_class_name(self):
        from envrax.env import EnvConfig
        from envrax.vec_env import VecEnv
        from tests.test_multi_vec_env import _VecEnv

        v = VecEnv(_VecEnv(config=EnvConfig(max_steps=5)), 4)
        assert v.name == "_VecEnv"

    def test_vec_env_render_slot(self):
        """render_slot delegates to inner env's render; inner has no render
        → NotImplementedError is acceptable."""
        from envrax.env import EnvConfig
        from envrax.vec_env import VecEnv
        from tests.test_multi_vec_env import _VecEnv

        v = VecEnv(_VecEnv(config=EnvConfig(max_steps=5)), 4)
        _, state = v.reset(jax.random.key(0))
        with pytest.raises(NotImplementedError):
            v.render_slot(state, 0)
