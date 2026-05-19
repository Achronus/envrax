import chex
import jax
import jax.numpy as jnp
import pytest

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


def _build_multi() -> MultiVecEnv:
    return MultiVecEnv({
        "vec": VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
        "pix": VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
    })


class TestMultiVecEnv:
    def test_n_envs(self):
        assert _build_multi().n_envs == 2

    def test_total_slots(self):
        multi = MultiVecEnv({
            "vec": VecEnv(_VecEnv(config=_CONFIG), 4),
            "pix": VecEnv(_PixEnv(config=_CONFIG), 8),
        })
        assert multi.total_slots == 12

    def test_slots_per_env(self):
        multi = MultiVecEnv({
            "vec": VecEnv(_VecEnv(config=_CONFIG), 4),
            "pix": VecEnv(_PixEnv(config=_CONFIG), 8),
        })
        assert multi.slots_per_env == {"vec": 4, "pix": 8}

    def test_envs_property(self):
        envs = {
            "vec": VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            "pix": VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        }
        multi = MultiVecEnv(envs)
        assert set(multi.envs.keys()) == {"vec", "pix"}

    def test_env_keys_preserves_insertion_order(self):
        multi = MultiVecEnv({
            "z": VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            "a": VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        })
        assert multi.env_keys == ["z", "a"]

    def test_reset_returns_dicts(self):
        obs, states = _build_multi().reset(_RNG)
        assert isinstance(obs, dict)
        assert isinstance(states, dict)
        assert set(obs.keys()) == {"vec", "pix"}
        assert set(states.keys()) == {"vec", "pix"}

    def test_reset_obs_have_batch_dim(self):
        obs, _ = _build_multi().reset(_RNG)
        assert obs["vec"].shape == (_N_ENVS, 4)
        assert obs["pix"].shape == (_N_ENVS, 4, 4, 3)

    def test_step_returns_correct_structure(self):
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        actions = {
            "vec": jnp.zeros(_N_ENVS, dtype=jnp.int32),
            "pix": jnp.zeros(_N_ENVS, dtype=jnp.int32),
        }
        obs, _, rewards, dones, _ = multi.step(states, actions)
        assert set(obs.keys()) == {"vec", "pix"}
        assert rewards["vec"].shape == (_N_ENVS,)
        assert dones["pix"].shape == (_N_ENVS,)

    def test_state_is_pytree(self):
        """States dict should be a proper JAX pytree."""
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        leaves = jax.tree.leaves(states)
        assert len(leaves) > 0
        assert all(isinstance(x, jax.Array) for x in leaves)

    def test_step_dispatches_inside_one_jit(self):
        """Both inner steps should appear in a single jaxpr (no per-call dispatch)."""
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        actions = {
            "vec": jnp.zeros(_N_ENVS, dtype=jnp.int32),
            "pix": jnp.zeros(_N_ENVS, dtype=jnp.int32),
        }
        jaxpr = jax.make_jaxpr(multi._step_impl)(states, actions)
        repr_str = str(jaxpr)
        # Both env types' dtypes should appear in the same jaxpr — `_VecEnv`
        # contributes f32 ops, `_PixEnv` contributes u8 ops.
        assert "f32" in repr_str
        assert "u8" in repr_str

    def test_single_observation_spaces(self):
        spaces = _build_multi().single_observation_spaces
        assert spaces["vec"].shape == (4,)
        assert spaces["pix"].shape == (4, 4, 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            MultiVecEnv({})

    def test_step_mismatched_state_keys_raises(self):
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        actions = {
            "vec": jnp.zeros(_N_ENVS, dtype=jnp.int32),
            "pix": jnp.zeros(_N_ENVS, dtype=jnp.int32),
        }
        states_bad = {"vec": states["vec"]}  # missing "pix"
        with pytest.raises(ValueError, match="`states` keys"):
            multi.step(states_bad, actions)

    def test_step_mismatched_action_keys_raises(self):
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        actions_bad = {"vec": jnp.zeros(_N_ENVS, dtype=jnp.int32)}  # missing "pix"
        with pytest.raises(ValueError, match="`actions` keys"):
            multi.step(states, actions_bad)

    def test_compile(self):
        multi = _build_multi()
        multi.compile(progress=False)
        obs, _ = multi.reset(_RNG)
        assert obs["vec"].shape == (_N_ENVS, 4)
        assert obs["pix"].shape == (_N_ENVS, 4, 4, 3)

    def test_repr(self):
        multi = MultiVecEnv({
            "vec": VecEnv(_VecEnv(config=_CONFIG), 4),
            "pix": VecEnv(_PixEnv(config=_CONFIG), 8),
        })
        r = repr(multi)
        assert "MultiVecEnv" in r
        assert "total_slots=12" in r

    def test_len_equals_n_envs(self):
        assert len(_build_multi()) == 2

    def test_slot_state(self):
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        single = multi.slot_state(states, "vec", 0)
        assert single.step.shape == ()


class TestMultiVecEnvAutoKeying:
    def test_list_input_uses_env_name_as_key(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        assert multi.env_keys == ["_VecEnv", "_PixEnv"]

    def test_list_input_suffixes_duplicates(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_PixEnv(config=_CONFIG), _N_ENVS),
        ])
        assert multi.env_keys == ["_VecEnv_0", "_VecEnv_1", "_PixEnv"]

    def test_list_input_all_duplicates_all_suffixed(self):
        multi = MultiVecEnv([
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
        ])
        assert multi.env_keys == ["_VecEnv_0", "_VecEnv_1", "_VecEnv_2"]

    def test_dict_input_preserves_explicit_keys(self):
        multi = MultiVecEnv({
            "task_a": VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
            "task_b": VecEnv(_VecEnv(config=_CONFIG), _N_ENVS),
        })
        assert multi.env_keys == ["task_a", "task_b"]


class TestMultiVecEnvHelpers:
    def test_single_observation_shapes(self):
        assert _build_multi().single_observation_shapes == {
            "vec": (4,),
            "pix": (4, 4, 3),
        }

    def test_single_action_shapes(self):
        assert _build_multi().single_action_shapes == {"vec": (), "pix": ()}

    def test_single_observation_sizes(self):
        assert _build_multi().single_observation_sizes == {"vec": 4, "pix": 48}

    def test_single_action_sizes(self):
        # prod(()) == 1
        assert _build_multi().single_action_sizes == {"vec": 1, "pix": 1}

    def test_single_observation_dtypes(self):
        assert _build_multi().single_observation_dtypes == {
            "vec": jnp.float32,
            "pix": jnp.uint8,
        }

    def test_single_action_dtypes(self):
        assert _build_multi().single_action_dtypes == {
            "vec": jnp.int32,
            "pix": jnp.int32,
        }

    def test_pad_dims_returns_max_action_and_observation(self):
        action, observation = _build_multi().pad_dims()
        assert action == 1
        assert observation == 48

    def test_pad_dims_returns_tuple_of_ints(self):
        result = _build_multi().pad_dims()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, int) for x in result)
