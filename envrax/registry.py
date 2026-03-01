# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Dict, List, Tuple, Type

from envrax.base import EnvParams, JaxEnv

_REGISTRY: Dict[str, Tuple[Type[JaxEnv], EnvParams]] = {}


def register(name: str, env_class: Type[JaxEnv], default_params: EnvParams) -> None:
    """
    Register an environment class under a given name.

    Called on package import by downstream packages (atarax, proxen, labrax)
    to make their environments available via `envrax.make()`.

    Parameters
    ----------
    name : str
        Unique environment name (e.g. ``"Breakout-v0"``).
    env_class : Type[JaxEnv]
        The environment class to register.
    default_params : EnvParams
        Default parameters for this environment.

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(
            f"Environment {name!r} is already registered. "
            "Use a unique name or unregister the existing entry first."
        )
    _REGISTRY[name] = (env_class, default_params)


def make(name: str, **param_overrides) -> Tuple[JaxEnv, EnvParams]:
    """
    Create an environment instance by name.

    Works for any installed package that registers its environments on import
    (atarax, proxen, labrax all do this).

    Parameters
    ----------
    name : str
        Registered environment name (e.g. ``"Breakout-v0"``).
    **param_overrides
        Keyword arguments forwarded to ``EnvParams.replace()`` to override
        individual default parameters.

    Returns
    -------
    env : JaxEnv
        Instantiated environment.
    params : EnvParams
        Resolved parameters (defaults merged with any overrides).

    Raises
    ------
    ValueError
        If ``name`` is not registered.

    Examples
    --------
    >>> import atarax   # registers atarax envs into envrax on import
    >>> env, params = envrax.make("Breakout-v0", max_steps=2000)
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(f"Unknown environment: {name!r}. Available: {available}")
    env_class, default_params = _REGISTRY[name]
    params = (
        default_params.__replace__(**param_overrides)
        if param_overrides
        else default_params
    )
    return env_class(), params


def registered_names() -> List[str]:
    """Return a sorted list of all registered environment names."""
    return sorted(_REGISTRY)
