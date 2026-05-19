import chex
import jax
import jax.numpy as jnp
import pytest

from envrax.env import EnvConfig, EnvState, JaxEnv
from envrax.multi_env import MultiEnv
from envrax.spaces import Box, Discrete

# ---------------------------------------------------------------------------
# Two minimal envs with different obs shapes for heterogeneity testing
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
        s = state.__replace__(step=state.step + 1)
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
        s = state.__replace__(step=state.step + 1)
        return jnp.full((4, 4, 3), 64, jnp.uint8), s, jnp.float32(1.0), jnp.bool_(False), {}


_RNG = jax.random.key(0)
_CONFIG = EnvConfig(max_steps=10)


def _build_multi() -> MultiEnv:
    return MultiEnv({
        "vec": _VecEnv(config=_CONFIG),
        "pix": _PixEnv(config=_CONFIG),
    })


class TestMultiEnv:
    def test_n_envs(self):
        assert _build_multi().n_envs == 2
        assert len(_build_multi()) == 2

    def test_envs_property_is_dict(self):
        multi = _build_multi()
        assert isinstance(multi.envs, dict)
        assert set(multi.envs.keys()) == {"vec", "pix"}

    def test_env_keys_preserves_insertion_order(self):
        multi = MultiEnv({
            "z": _VecEnv(config=_CONFIG),
            "a": _PixEnv(config=_CONFIG),
        })
        assert multi.env_keys == ["z", "a"]

    def test_observation_spaces(self):
        spaces = _build_multi().observation_spaces
        assert spaces["vec"].shape == (4,)
        assert spaces["pix"].shape == (4, 4, 3)

    def test_action_spaces(self):
        spaces = _build_multi().action_spaces
        assert spaces["vec"].n == 2
        assert spaces["pix"].n == 3

    def test_reset_returns_dicts(self):
        obs, states = _build_multi().reset(_RNG)
        assert isinstance(obs, dict)
        assert isinstance(states, dict)
        assert set(obs.keys()) == {"vec", "pix"}
        assert set(states.keys()) == {"vec", "pix"}

    def test_reset_obs_shapes_match_spaces(self):
        obs, _ = _build_multi().reset(_RNG)
        assert obs["vec"].shape == (4,)
        assert obs["pix"].shape == (4, 4, 3)

    def test_reset_deterministic_same_seed(self):
        multi = MultiEnv({"only": _VecEnv(config=_CONFIG)})
        obs1, _ = multi.reset(_RNG)
        obs2, _ = multi.reset(_RNG)
        assert jnp.array_equal(obs1["only"], obs2["only"])

    def test_step_returns_correct_structure(self):
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        actions = {"vec": jnp.int32(0), "pix": jnp.int32(0)}
        obs, new_states, rewards, dones, infos = multi.step(states, actions)
        assert set(obs.keys()) == {"vec", "pix"}
        assert set(rewards.keys()) == {"vec", "pix"}
        assert set(dones.keys()) == {"vec", "pix"}
        assert set(infos.keys()) == {"vec", "pix"}

    def test_step_obs_shapes(self):
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        actions = {"vec": jnp.int32(0), "pix": jnp.int32(0)}
        obs, _, _, _, _ = multi.step(states, actions)
        assert obs["vec"].shape == (4,)
        assert obs["pix"].shape == (4, 4, 3)

    def test_per_env_access_via_envs_dict(self):
        multi = _build_multi()
        single = multi.envs["pix"]
        obs, state = single.reset(_RNG)
        assert obs.shape == (4, 4, 3)
        assert int(state.step) == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            MultiEnv([])

    def test_step_mismatched_state_keys_raises(self):
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        actions = {"vec": jnp.int32(0), "pix": jnp.int32(0)}
        states_bad = {"vec": states["vec"]}  # missing "pix"
        with pytest.raises(ValueError, match="`states` keys"):
            multi.step(states_bad, actions)

    def test_step_mismatched_action_keys_raises(self):
        multi = _build_multi()
        _, states = multi.reset(_RNG)
        actions_bad = {"vec": jnp.int32(0)}  # missing "pix"
        with pytest.raises(ValueError, match="`actions` keys"):
            multi.step(states, actions_bad)

    def test_compile_with_jit_wrapped_envs(self):
        from envrax.wrappers import JitWrapper

        multi = MultiEnv({
            "vec": JitWrapper(_VecEnv(config=_CONFIG), cache_dir=None, pre_warm=False),
            "pix": JitWrapper(_PixEnv(config=_CONFIG), cache_dir=None, pre_warm=False),
        })
        multi.compile(progress=False)
        obs, _ = multi.reset(_RNG)
        assert obs["vec"].shape == (4,)
        assert obs["pix"].shape == (4, 4, 3)

    def test_compile_skips_non_jit_envs(self):
        multi = MultiEnv({"only": _VecEnv(config=_CONFIG)})
        multi.compile(progress=False)  # should not raise

    def test_repr(self):
        r = repr(_build_multi())
        assert "MultiEnv" in r
        assert "vec" in r
        assert "pix" in r


class TestMultiEnvAutoKeying:
    def test_list_input_uses_env_name_as_key(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG), _PixEnv(config=_CONFIG)])
        assert multi.env_keys == ["_VecEnv", "_PixEnv"]

    def test_list_input_suffixes_duplicates(self):
        multi = MultiEnv([
            _VecEnv(config=_CONFIG),
            _VecEnv(config=_CONFIG),
            _PixEnv(config=_CONFIG),
        ])
        assert multi.env_keys == ["_VecEnv_0", "_VecEnv_1", "_PixEnv"]

    def test_list_input_all_duplicates_all_suffixed(self):
        multi = MultiEnv([_VecEnv(config=_CONFIG)] * 3)
        assert multi.env_keys == ["_VecEnv_0", "_VecEnv_1", "_VecEnv_2"]

    def test_dict_input_preserves_explicit_keys(self):
        multi = MultiEnv({
            "task_a": _VecEnv(config=_CONFIG),
            "task_b": _VecEnv(config=_CONFIG),
        })
        assert multi.env_keys == ["task_a", "task_b"]

    def test_wrapper_delegates_name(self):
        """JitWrapper.name should delegate to inner env, not return 'JitWrapper'."""
        from envrax.wrappers import JitWrapper

        wrapped = JitWrapper(_VecEnv(config=_CONFIG), cache_dir=None, pre_warm=False)
        assert wrapped.name == "_VecEnv"

        multi = MultiEnv([wrapped])
        assert multi.env_keys == ["_VecEnv"]


class TestMultiEnvHelpers:
    def test_observation_shapes(self):
        assert _build_multi().observation_shapes == {"vec": (4,), "pix": (4, 4, 3)}

    def test_action_shapes(self):
        assert _build_multi().action_shapes == {"vec": (), "pix": ()}

    def test_observation_sizes(self):
        assert _build_multi().observation_sizes == {"vec": 4, "pix": 48}

    def test_action_sizes(self):
        assert _build_multi().action_sizes == {"vec": 1, "pix": 1}

    def test_observation_dtypes(self):
        assert _build_multi().observation_dtypes == {"vec": jnp.float32, "pix": jnp.uint8}

    def test_action_dtypes(self):
        assert _build_multi().action_dtypes == {"vec": jnp.int32, "pix": jnp.int32}

    def test_pad_dims_returns_max_action_and_observation(self):
        action, observation = _build_multi().pad_dims()
        assert action == 1
        assert observation == 48

    def test_pad_dims_returns_tuple_of_ints(self):
        result = _build_multi().pad_dims()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, int) for x in result)
