from dataclasses import dataclass, field
from typing import List

import chex
import jax
import jax.numpy as jnp
import pytest

from envrax.env import EnvConfig, EnvState, JaxEnv
from envrax.make import make
from envrax.registry import (
    _REGISTRY,
    get_spec,
    register,
    register_suite,
    registered_names,
)
from envrax.spaces import Box, Discrete
from envrax.suite import EnvSet, EnvSpec, EnvSuite

# ---------------------------------------------------------------------------
# Minimal concrete env for testing
# ---------------------------------------------------------------------------


@chex.dataclass
class _DummyState(EnvState):
    pass


class _DummyEnv(JaxEnv[Box, Discrete, _DummyState, EnvConfig]):
    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=(4,), dtype=jnp.float32)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=2)

    def reset(self, rng: chex.PRNGKey):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        state = _DummyState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, state, action):
        obs = jnp.zeros((4,), dtype=jnp.float32)
        new_state = state.__replace__(step=state.step + 1)
        reward = jnp.float32(0.0)
        done = new_state.step >= self.config.max_steps
        return obs, new_state, reward, done, {}


class TestRegistry:
    def setup_method(self):
        # Clean slate for each test by removing our test env if registered
        _REGISTRY.pop("DummyEnv-v0", None)

    def teardown_method(self):
        _REGISTRY.pop("DummyEnv-v0", None)

    def test_register_and_make(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        env, config = make("DummyEnv-v0", jit_compile=False)
        assert isinstance(env, _DummyEnv)
        assert config.max_steps == 1000

    def test_make_with_custom_config(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        env, config = make(
            "DummyEnv-v0", config=EnvConfig(max_steps=500), jit_compile=False
        )
        assert config.max_steps == 500

    def test_make_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            make("DoesNotExist-v0", jit_compile=False)

    def test_register_duplicate_raises(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        with pytest.raises(ValueError, match="already registered"):
            register("DummyEnv-v0", _DummyEnv, EnvConfig())

    def test_env_reset_step(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        env, _ = make("DummyEnv-v0", jit_compile=False)
        rng = jax.random.key(0)
        obs, state = env.reset(rng)
        assert obs.shape == (4,)
        obs2, state2, reward, done, info = env.step(state, jnp.int32(0))
        assert obs2.shape == (4,)
        assert int(state2.step) == 1

    def test_register_with_suite_tag(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig(), suite="dummy-suite")
        spec = get_spec("DummyEnv-v0")
        assert spec.suite == "dummy-suite"

    def test_registered_names_returns_keys(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        assert "DummyEnv-v0" in registered_names()


# ---------------------------------------------------------------------------
# EnvSpec / EnvSuite / register_suite tests
# ---------------------------------------------------------------------------


@dataclass
class _DummySuite(EnvSuite):
    """Concrete EnvSuite for testing register_suite()."""

    prefix: str = "dummy"
    category: str = "Dummy"
    version: str = "v0"
    required_packages: List[str] = field(default_factory=list)
    specs: List[EnvSpec] = field(
        default_factory=lambda: [
            EnvSpec("alpha", _DummyEnv, EnvConfig(max_steps=100)),
            EnvSpec("beta", _DummyEnv, EnvConfig(max_steps=200)),
        ]
    )

    def get_name(self, name: str, version: str | None = None) -> str:
        ver = version if version is not None else self.version
        return f"{self.prefix}/{name}-{ver}"


class TestEnvSpec:
    def test_fields(self):
        spec = EnvSpec(
            name="dummy/alpha-v0",
            env_class=_DummyEnv,
            default_config=EnvConfig(max_steps=42),
            suite="Dummy",
        )
        assert spec.name == "dummy/alpha-v0"
        assert spec.env_class is _DummyEnv
        assert spec.default_config.max_steps == 42
        assert spec.suite == "Dummy"

    def test_default_suite_is_empty(self):
        spec = EnvSpec("alpha", _DummyEnv, EnvConfig())
        assert spec.suite == ""

    def test_is_frozen(self):
        spec = EnvSpec("alpha", _DummyEnv, EnvConfig())
        with pytest.raises(Exception):
            spec.name = "beta"  # type: ignore[misc]


class TestEnvSuite:
    def test_envs_property_derives_from_specs(self):
        suite = _DummySuite()
        assert suite.envs == ["alpha", "beta"]

    def test_n_envs(self):
        suite = _DummySuite()
        assert suite.n_envs == 2
        assert len(suite) == 2

    def test_get_name(self):
        suite = _DummySuite()
        assert suite.get_name("alpha") == "dummy/alpha-v0"
        assert suite.get_name("alpha", version="v1") == "dummy/alpha-v1"

    def test_all_names(self):
        suite = _DummySuite()
        assert suite.all_names() == ["dummy/alpha-v0", "dummy/beta-v0"]

    def test_iter_yields_canonical_names(self):
        suite = _DummySuite()
        assert list(suite) == ["dummy/alpha-v0", "dummy/beta-v0"]

    def test_contains(self):
        suite = _DummySuite()
        assert "alpha" in suite
        assert "missing" not in suite

    def test_slicing_returns_same_class(self):
        suite = _DummySuite()
        sliced = suite[:1]
        assert isinstance(sliced, _DummySuite)
        assert sliced.n_envs == 1
        assert sliced.envs == ["alpha"]

    def test_index_returns_single_spec_suite(self):
        suite = _DummySuite()
        single = suite[1]
        assert single.envs == ["beta"]


class TestRegisterSuite:
    def setup_method(self):
        for name in (
            "dummy/alpha-v0",
            "dummy/beta-v0",
            "dummy/alpha-v1",
            "dummy/beta-v1",
        ):
            _REGISTRY.pop(name, None)

    def teardown_method(self):
        for name in (
            "dummy/alpha-v0",
            "dummy/beta-v0",
            "dummy/alpha-v1",
            "dummy/beta-v1",
        ):
            _REGISTRY.pop(name, None)

    def test_registers_all_specs(self):
        register_suite(_DummySuite())
        assert "dummy/alpha-v0" in registered_names()
        assert "dummy/beta-v0" in registered_names()

    def test_registered_specs_use_canonical_name(self):
        register_suite(_DummySuite())
        spec = get_spec("dummy/alpha-v0")
        assert spec.name == "dummy/alpha-v0"

    def test_propagates_suite_category(self):
        register_suite(_DummySuite())
        spec = get_spec("dummy/alpha-v0")
        assert spec.suite == "Dummy"

    def test_preserves_per_env_configs(self):
        register_suite(_DummySuite())
        assert get_spec("dummy/alpha-v0").default_config.max_steps == 100
        assert get_spec("dummy/beta-v0").default_config.max_steps == 200

    def test_make_after_register_suite(self):
        register_suite(_DummySuite())
        env, config = make("dummy/alpha-v0", jit_compile=False)
        assert isinstance(env, _DummyEnv)
        assert config.max_steps == 100

    def test_version_override(self):
        register_suite(_DummySuite(), version="v1")
        assert "dummy/alpha-v1" in registered_names()
        assert "dummy/alpha-v0" not in registered_names()

    def test_duplicate_suite_registration_raises(self):
        register_suite(_DummySuite())
        with pytest.raises(ValueError, match="already registered"):
            register_suite(_DummySuite())


class TestGetSpec:
    def setup_method(self):
        _REGISTRY.pop("DummyEnv-v0", None)

    def teardown_method(self):
        _REGISTRY.pop("DummyEnv-v0", None)

    def test_returns_spec(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig(), suite="dummy")
        spec = get_spec("DummyEnv-v0")
        assert isinstance(spec, EnvSpec)
        assert spec.name == "DummyEnv-v0"
        assert spec.env_class is _DummyEnv
        assert spec.suite == "dummy"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            get_spec("DoesNotExist-v0")


# ---------------------------------------------------------------------------
# EnvSet tests
# ---------------------------------------------------------------------------


class TestEnvSet:
    def test_n_envs_aggregates(self):
        s = EnvSet(_DummySuite(), _DummySuite())
        assert s.n_envs == 4
        assert len(s) == 4

    def test_all_names_aggregates(self):
        s = EnvSet(_DummySuite())
        assert s.all_names() == ["dummy/alpha-v0", "dummy/beta-v0"]

    def test_categories(self):
        s = EnvSet(_DummySuite())
        assert s.env_categories() == {"Dummy": 2}

    def test_add_combines(self):
        s1 = EnvSet(_DummySuite())
        s2 = EnvSet(_DummySuite())
        combined = s1 + s2
        assert combined.n_envs == 4
        assert len(combined.suites) == 2
