from dataclasses import dataclass, field
from typing import List

import chex
import jax
import jax.numpy as jnp
import pytest

from envrax.env import EnvConfig, EnvState, JaxEnv
from envrax.error import MissingPackageError
from envrax.make import make
from envrax.registry import (
    _REGISTRY,
    get_spec,
    register,
    register_suite,
    registered_names,
)
from envrax.spaces import Box, Discrete
from envrax.suite import EnvSet, EnvSpec, EnvSuite, _RegisteredSuite

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
        env = make("DummyEnv-v0", jit_compile=False)
        assert isinstance(env, _DummyEnv)
        assert env.config.max_steps == 1000

    def test_make_with_custom_config(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        env = make(
            "DummyEnv-v0", config=EnvConfig(max_steps=500), jit_compile=False
        )
        assert env.config.max_steps == 500

    def test_make_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            make("DoesNotExist-v0", jit_compile=False)

    def test_register_duplicate_raises(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        with pytest.raises(ValueError, match="already registered"):
            register("DummyEnv-v0", _DummyEnv, EnvConfig())

    def test_env_reset_step(self):
        register("DummyEnv-v0", _DummyEnv, EnvConfig())
        env = make("DummyEnv-v0", jit_compile=False)
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

    def test_default_get_name_uses_prefix_format(self):
        # Bare EnvSuite (no override) — exercises the base class default
        suite = EnvSuite(
            prefix="bare",
            category="Bare",
            version="v0",
            specs=[EnvSpec("alpha", _DummyEnv, EnvConfig())],
        )
        assert suite.get_name("alpha") == "bare/alpha-v0"
        assert suite.get_name("alpha", version="v2") == "bare/alpha-v2"

    def test_check_returns_per_package_status(self):
        suite = EnvSuite(
            prefix="x",
            category="X",
            required_packages=["jax", "definitely_not_a_real_package_xyz"],
        )
        status = suite.check()
        assert status["jax"] is True
        assert status["definitely_not_a_real_package_xyz"] is False

    def test_is_available_true_when_no_packages(self):
        suite = EnvSuite(prefix="x", category="X", required_packages=[])
        assert suite.is_available() is True

    def test_is_available_true_when_all_installed(self):
        suite = EnvSuite(prefix="x", category="X", required_packages=["jax"])
        assert suite.is_available() is True

    def test_is_available_false_when_any_missing(self):
        suite = EnvSuite(
            prefix="x",
            category="X",
            required_packages=["jax", "definitely_not_a_real_package_xyz"],
        )
        assert suite.is_available() is False


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
        env = make("dummy/alpha-v0", jit_compile=False)
        assert isinstance(env, _DummyEnv)
        assert env.config.max_steps == 100

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

    def test_iter_yields_canonical_names_across_suites(self):
        s = EnvSet(_DummySuite(), _DummySuite())
        assert list(s) == [
            "dummy/alpha-v0",
            "dummy/beta-v0",
            "dummy/alpha-v0",
            "dummy/beta-v0",
        ]

    def test_repr_includes_suite_classes_and_total(self):
        s = EnvSet(_DummySuite(), _DummySuite())
        r = repr(s)
        assert "EnvSet" in r
        assert "_DummySuite" in r
        assert "total=4" in r

    def test_verify_packages_passes_when_all_installed(self):
        s = EnvSet(
            EnvSuite(prefix="a", category="A", required_packages=["jax"]),
            EnvSuite(prefix="b", category="B", required_packages=[]),
        )
        s.verify_packages()  # should not raise

    def test_verify_packages_raises_with_missing(self):
        s = EnvSet(
            EnvSuite(
                prefix="a",
                category="MissingSuite",
                required_packages=["definitely_not_a_real_package_xyz"],
            ),
        )
        with pytest.raises(MissingPackageError, match="MissingSuite"):
            s.verify_packages()

    def test_verify_packages_lists_each_missing_suite(self):
        s = EnvSet(
            EnvSuite(
                prefix="a",
                category="Alpha",
                required_packages=["definitely_not_a_real_package_xyz"],
            ),
            EnvSuite(
                prefix="b",
                category="Beta",
                required_packages=["jax"],  # installed — should not appear
            ),
        )
        with pytest.raises(MissingPackageError) as exc_info:
            s.verify_packages()
        msg = str(exc_info.value)
        assert "Alpha" in msg
        assert "Beta" not in msg


class _BoxEnv(JaxEnv[Box, Box, _DummyState, EnvConfig]):
    """Env with Box obs/action spaces — for pad_dims success-path tests."""

    OBS_SHAPE: tuple = (4,)
    ACT_SHAPE: tuple = (2,)

    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=self.OBS_SHAPE, dtype=jnp.float32)

    @property
    def action_space(self) -> Box:
        return Box(low=-1, high=1, shape=self.ACT_SHAPE, dtype=jnp.float32)

    def reset(self, rng: chex.PRNGKey):
        obs = jnp.zeros(self.OBS_SHAPE, dtype=jnp.float32)
        state = _DummyState(rng=rng, step=jnp.int32(0), done=jnp.bool_(False))
        return obs, state

    def step(self, state, action):
        obs = jnp.zeros(self.OBS_SHAPE, dtype=jnp.float32)
        new_state = state.__replace__(step=state.step + 1)
        return obs, new_state, jnp.float32(0.0), jnp.bool_(False), {}


class _BigBoxEnv(_BoxEnv):
    OBS_SHAPE = (8, 8, 3)  # flat = 192
    ACT_SHAPE = (5,)


class TestEnvSetFromNames:
    _KEYS = ("dummy/alpha-v0", "dummy/beta-v0", "other/gamma-v0")

    def setup_method(self):
        for k in self._KEYS:
            _REGISTRY.pop(k, None)

    def teardown_method(self):
        for k in self._KEYS:
            _REGISTRY.pop(k, None)

    def test_groups_by_category(self):
        register_suite(_DummySuite())  # category="Dummy"
        register("other/gamma-v0", _DummyEnv, EnvConfig(), suite="Other")
        s = EnvSet.from_names(
            ["dummy/alpha-v0", "dummy/beta-v0", "other/gamma-v0"]
        )
        assert s.env_categories() == {"Dummy": 2, "Other": 1}

    def test_iter_yields_canonical_names(self):
        register("other/gamma-v0", _DummyEnv, EnvConfig(), suite="Other")
        s = EnvSet.from_names(["other/gamma-v0"])
        assert list(s) == ["other/gamma-v0"]

    def test_all_names_yields_canonical_names(self):
        register_suite(_DummySuite())
        s = EnvSet.from_names(["dummy/alpha-v0", "dummy/beta-v0"])
        assert s.all_names() == ["dummy/alpha-v0", "dummy/beta-v0"]

    def test_suite_type_is_registered_suite(self):
        register("other/gamma-v0", _DummyEnv, EnvConfig(), suite="Other")
        s = EnvSet.from_names(["other/gamma-v0"])
        assert isinstance(s.suites[0], _RegisteredSuite)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            EnvSet.from_names(["not/registered-v0"])


class TestPadDims:
    _KEYS = (
        "padtest/small-v0",
        "padtest/big-v0",
        "padtest/discrete-v0",
        "padtest2/lone-v0",
    )

    def setup_method(self):
        for k in self._KEYS:
            _REGISTRY.pop(k, None)

    def teardown_method(self):
        for k in self._KEYS:
            _REGISTRY.pop(k, None)

    def _box_suite(self) -> EnvSuite:
        register("padtest/small-v0", _BoxEnv, EnvConfig(), suite="Pad")
        register("padtest/big-v0", _BigBoxEnv, EnvConfig(), suite="Pad")
        return EnvSuite(
            prefix="padtest",
            category="Pad",
            version="v0",
            specs=[
                EnvSpec("small", _BoxEnv, EnvConfig()),
                EnvSpec("big", _BigBoxEnv, EnvConfig()),
            ],
        )

    def test_suite_returns_tuple_of_two_ints(self):
        suite = self._box_suite()
        result = suite.pad_dims()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, int) for x in result)

    def test_suite_takes_max_across_envs(self):
        suite = self._box_suite()
        action, observation = suite.pad_dims()
        assert action == 5  # max(2, 5)
        assert observation == 192  # max(4, 8*8*3)

    def test_suite_uses_flat_size_for_multidim_shapes(self):
        register("padtest/big-v0", _BigBoxEnv, EnvConfig(), suite="Pad")
        suite = EnvSuite(
            prefix="padtest",
            category="Pad",
            version="v0",
            specs=[EnvSpec("big", _BigBoxEnv, EnvConfig())],
        )
        _, observation = suite.pad_dims()
        assert observation == 8 * 8 * 3

    def test_suite_raises_for_space_without_shape(self):
        register("padtest/discrete-v0", _DummyEnv, EnvConfig(), suite="Pad")
        suite = EnvSuite(
            prefix="padtest",
            category="Pad",
            version="v0",
            specs=[EnvSpec("discrete", _DummyEnv, EnvConfig())],
        )
        with pytest.raises(TypeError, match="has no `.shape`"):
            suite.pad_dims()

    def test_envset_aggregates_across_suites(self):
        register("padtest/small-v0", _BoxEnv, EnvConfig(), suite="A")
        register("padtest2/lone-v0", _BigBoxEnv, EnvConfig(), suite="B")
        s = EnvSet.from_names(["padtest/small-v0", "padtest2/lone-v0"])
        action, observation = s.pad_dims()
        assert action == 5
        assert observation == 192
