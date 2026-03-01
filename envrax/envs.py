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

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Dict, Iterator, List, Self, Union

from envrax.error import MissingPackageError


@dataclass
class EnvGroup:
    """
    Base environment group dataclass.

    A named, versioned collection of environment IDs belonging to the same
    suite (e.g. all Atari games). Subclasses override `get_name` to
    produce the canonical ID string for each entry.

    Attributes
    ----------
    prefix : str
        Namespace prefix for environment names (e.g. `"atari"`).
    category : str
        Human-readable category label (e.g. `"Atari"`).
    version : str
        Version suffix applied by `get_name` (e.g. `"v0"`).
    required_packages : List[str]
        Python packages that must be importable for this group to work.
    envs : List[str]
        Short environment names stored in this group (e.g. `"breakout"`).
    """

    prefix: str = ""
    category: str = ""
    version: str = "v0"
    required_packages: List[str] = field(default_factory=list)
    envs: List[str] = field(default_factory=list)

    @property
    def n_envs(self) -> int:
        """Number of environments in this group."""
        return len(self.envs)

    def get_name(self, env: str, version: str | None = None) -> str:
        """
        Return the canonical ID string for a single environment.

        Parameters
        ----------
        env : str
            Short environment name as stored in `self.envs`.
        version : str (optional)
            Override the group's default version suffix.

        Returns
        -------
        name : str
            Full environment ID (e.g. `"atari/breakout-v0"`).
        """
        raise NotImplementedError("Subclasses must implement get_name")

    def all_names(self, version: str | None = None) -> List[str]:
        """
        Return canonical IDs for every environment in this group.

        Parameters
        ----------
        version : str (optional)
            Override the group's default version suffix.

        Returns
        -------
        names : List[str]
            One ID string per environment.
        """
        return [self.get_name(env, version) for env in self.envs]

    def __contains__(self, env: str) -> bool:
        """Return `True` if *env* is in `self.envs`."""
        return env in self.envs

    def __getitem__(self, key: Union[int, slice]) -> "EnvGroup":
        """
        Return a new group containing only the selected environment(s).

        Parameters
        ----------
        key : int | slice
            Index or slice into `self.envs`.

        Returns
        -------
        group : EnvGroup
            Same class as `self`, with the subset of environments.
        """
        if isinstance(key, int):
            selected = [self.envs[key]]
        else:
            selected = self.envs[key]

        return self.__class__(prefix=self.prefix, envs=selected)

    def __iter__(self) -> Iterator[str]:
        """Yield canonical ID strings for all environments."""
        for env in self.envs:
            yield self.get_name(env)

    def __len__(self) -> int:
        """Return number of environments."""
        return len(self.envs)

    def check(self) -> Dict[str, bool]:
        """
        Check whether each required package is importable.

        Returns
        -------
        status : Dict[str, bool]
            Mapping of package name → installed flag.
        """
        return {pkg: find_spec(pkg) is not None for pkg in self.required_packages}

    def is_available(self) -> bool:
        """
        Return `True` if all required packages are installed.

        Returns
        -------
        available : bool
        """
        return all(self.check().values())


class EnvSet:
    """
    An ordered collection of :class:`EnvGroup` instances.

    Combines multiple groups into a single iterable that yields canonical
    environment ID strings. Supports slicing via the groups themselves and
    merging two `EnvSet` objects with `+`.

    Parameters
    ----------
    *groups : EnvGroup
        Variable number of environment groups to combine.

    Examples
    --------
    >>> env_set = EnvSet(ATARI_BASE, ATARI_EASY)
    >>> for env_id in env_set:
    ...     print(env_id)
    """

    def __init__(self, *groups: EnvGroup) -> None:
        self._groups: List[EnvGroup] = list(groups)

    @property
    def n_envs(self) -> int:
        """Total number of environments across all groups."""
        return sum(g.n_envs for g in self._groups)

    @property
    def groups(self) -> List[EnvGroup]:
        """List of environment groups in this set."""
        return self._groups

    def all_names(self, version: str | None = None) -> List[str]:
        """
        Return canonical IDs for every environment across all groups.

        Parameters
        ----------
        version : str (optional)
            Override the default version suffix for all groups.

        Returns
        -------
        names : List[str]
        """
        names: List[str] = []
        for group in self._groups:
            names.extend(group.all_names(version))
        return names

    def env_categories(self) -> Dict[str, int]:
        """
        Return a mapping of category name → environment count.

        Returns
        -------
        categories : Dict[str, int]
        """
        counts: Dict[str, int] = {}
        for g in self._groups:
            counts[g.category] = counts.get(g.category, 0) + g.n_envs
        return counts

    def __iter__(self) -> Iterator[str]:
        """Yield canonical ID strings from all groups in order."""
        for group in self._groups:
            yield from group

    def __len__(self) -> int:
        """Total number of environments."""
        return self.n_envs

    def __add__(self, other: Self) -> Self:
        """Merge two EnvSets into a new one."""
        return type(self)(*self._groups, *other._groups)

    def verify_packages(self) -> None:
        """
        Raise :exc:`MissingPackageError` if any group has uninstalled packages.

        Raises
        ------
        error : MissingPackageError
            If any group has missing required packages.
        """
        missing: Dict[str, List[str]] = {}
        for group in self._groups:
            status = group.check()
            not_installed = [pkg for pkg, ok in status.items() if not ok]
            if not_installed:
                missing[group.category] = not_installed

        if missing:
            lines = [f"  {cat}: {', '.join(pkgs)}" for cat, pkgs in missing.items()]
            raise MissingPackageError(
                "Missing required packages for environment groups:\n" + "\n".join(lines)
            )

    def __repr__(self) -> str:
        group_info = ", ".join(
            f"{g.__class__.__name__}({g.n_envs})" for g in self._groups
        )
        return f"EnvSet({group_info}, total={self.n_envs})"
