from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import pytest

from envrax.env import EnvConfig, EnvState, JaxEnv
from envrax.spaces import Box, Discrete

_RNG = jax.random.key(0)


class TestEnvState:
    def test_fields(self):
        state = EnvState(rng=_RNG, step=jnp.int32(0), done=jnp.bool_(False))
        assert int(state.step) == 0
        assert not bool(state.done)

    def test_replace(self):
        state = EnvState(rng=_RNG, step=jnp.int32(0), done=jnp.bool_(False))
        new_state = state.__replace__(step=jnp.int32(5))
        assert int(new_state.step) == 5
        assert not bool(new_state.done)


class TestEnvConfig:
    def test_default(self):
        config = EnvConfig()
        assert config.max_steps == 1000

    def test_custom(self):
        config = EnvConfig(max_steps=500)
        assert config.max_steps == 500

    def test_replace(self):
        config = EnvConfig()
        new_config = config.__replace__(max_steps=2000)
        assert new_config.max_steps == 2000


@chex.dataclass
class _DummyState(EnvState):
    pass


@chex.dataclass
class _CustomConfig(EnvConfig):
    gravity: float = 9.81
    reward_scale: float = 2.0


class _BaseDummyEnv(JaxEnv[Box, Discrete, _DummyState, EnvConfig]):
    """Pins `ConfigT = EnvConfig` (default config)."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, _DummyState]:
        return jnp.zeros(2), _DummyState(
            rng=rng, step=jnp.int32(0), done=jnp.bool_(False)
        )

    def step(
        self, state: _DummyState, action: chex.Array
    ) -> Tuple[chex.Array, _DummyState, chex.Array, chex.Array, Dict[str, Any]]:
        return jnp.zeros(2), state, jnp.float32(0.0), jnp.bool_(False), {}


class _CustomConfigEnv(JaxEnv[Box, Discrete, _DummyState, _CustomConfig]):
    """Pins `ConfigT = _CustomConfig`."""

    @property
    def observation_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(2,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, _DummyState]:
        return jnp.zeros(2), _DummyState(
            rng=rng, step=jnp.int32(0), done=jnp.bool_(False)
        )

    def step(
        self, state: _DummyState, action: chex.Array
    ) -> Tuple[chex.Array, _DummyState, chex.Array, chex.Array, Dict[str, Any]]:
        return jnp.zeros(2), state, jnp.float32(0.0), jnp.bool_(False), {}


class TestJaxEnvConfigResolution:
    def test_default_config_resolves_to_envconfig(self):
        env = _BaseDummyEnv()
        assert isinstance(env.config, EnvConfig)
        assert env.config.max_steps == 1000

    def test_custom_config_resolves_to_pinned_class(self):
        env = _CustomConfigEnv()
        assert isinstance(env.config, _CustomConfig)
        assert env.config.gravity == 9.81
        assert env.config.reward_scale == 2.0
        assert env.config.max_steps == 1000

    def test_explicit_config_overrides_default(self):
        env = _CustomConfigEnv(config=_CustomConfig(gravity=3.7, max_steps=500))
        assert isinstance(env.config, _CustomConfig)
        assert env.config.gravity == 3.7
        assert env.config.max_steps == 500

    def test_resolve_config_cls_classmethod(self):
        assert _BaseDummyEnv._resolve_config_cls() is EnvConfig
        assert _CustomConfigEnv._resolve_config_cls() is _CustomConfig

    def test_unpinned_subclass_raises(self):
        class _Unpinned(JaxEnv):  # type: ignore[type-arg]
            pass

        with pytest.raises(TypeError, match="does not pin a concrete type"):
            _Unpinned._resolve_config_cls()


class TestJaxEnvRenderDefault:
    def test_default_render_raises_not_implemented(self):
        env = _BaseDummyEnv()
        _, state = env.reset(_RNG)
        with pytest.raises(NotImplementedError, match="does not implement render"):
            env.render(state)
