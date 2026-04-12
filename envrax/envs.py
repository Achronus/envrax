from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Dict, Iterator, List, Self, Type, Union

from envrax.env import EnvConfig, JaxEnv
from envrax.error import MissingPackageError


@dataclass(frozen=True)
class EnvSpec:
    """
    Specification for a single environment — the unit of registration.

    Holds everything needed to instantiate a registered environment: its
    name, the class to construct, and the default config to pass. Used both
    as a definition-time artifact (inside `EnvSuite.specs`) and as the
    runtime value stored in the registry.

    Parameters
    ----------
    name : str
        Short name within an `EnvSuite` (e.g. `"cartpole"`) at definition
        time, or canonical ID (e.g. `"mjx/cartpole-v0"`) once registered.
    env_class : Type[JaxEnv]
        Environment class to instantiate.
    default_config : EnvConfig
        Default configuration passed to `env_class(config=...)`.
    suite : str
        Suite category tag (e.g. `"MuJoCo Playground"`). Populated by
        `register_suite` from the parent `EnvSuite.category`.
    """

    name: str
    env_class: Type[JaxEnv]
    default_config: EnvConfig
    suite: str = ""


@dataclass
class EnvSuite:
    """
    A named, versioned collection of environment specs from one suite.

    The `specs` list is the single source of truth for which environments
    the suite ships. The `envs` property derives short names from `specs`
    so that iteration, slicing, and display work without a parallel list.

    Subclasses pin `prefix`, `category`, `version`, `required_packages`
    and provide their `specs` via `default_factory`. They must also
    override `get_name` to produce canonical IDs (e.g. `"mjx/cartpole-v0"`).

    Parameters
    ----------
    prefix : str
        Namespace prefix for environment names (e.g. `"mjx"`).
    category : str
        Human-readable category label (e.g. `"MuJoCo Playground"`).
    version : str
        Version suffix applied by `get_name` (e.g. `"v0"`).
    required_packages : List[str]
        Python packages that must be importable for this suite to work.
    specs : List[EnvSpec]
        Environment specifications shipped by this suite.
    """

    prefix: str = ""
    category: str = ""
    version: str = "v0"
    required_packages: List[str] = field(default_factory=list)
    specs: List[EnvSpec] = field(default_factory=list)

    @property
    def envs(self) -> List[str]:
        """Short names of all environments in this suite, derived from specs."""
        return [s.name for s in self.specs]

    @property
    def n_envs(self) -> int:
        """Number of environments in this suite."""
        return len(self.specs)

    def get_name(self, name: str, version: str | None = None) -> str:
        """
        Return the canonical ID string for a single environment.

        Parameters
        ----------
        name : str
            Short environment name as stored on a spec.
        version : str (optional)
            Override the suite's default version suffix.

        Returns
        -------
        canonical : str
            Full environment ID (e.g. `"mjx/cartpole-v0"`).
        """
        raise NotImplementedError("Subclasses must implement get_name")

    def all_names(self, version: str | None = None) -> List[str]:
        """
        Return canonical IDs for every environment in this suite.

        Parameters
        ----------
        version : str (optional)
            Override the suite's default version suffix.

        Returns
        -------
        names : List[str]
            One canonical ID per spec.
        """
        return [self.get_name(s.name, version) for s in self.specs]

    def __contains__(self, name: str) -> bool:
        """Return `True` if a short name is in this suite's specs."""
        return any(s.name == name for s in self.specs)

    def __getitem__(self, key: Union[int, slice]) -> "EnvSuite":
        """
        Return a new suite containing only the selected spec(s).

        Parameters
        ----------
        key : int | slice
            Index or slice into `self.specs`.

        Returns
        -------
        suite : EnvSuite
            Same class as `self`, with the subset of specs.
        """
        if isinstance(key, int):
            selected = [self.specs[key]]
        else:
            selected = self.specs[key]

        return self.__class__(
            prefix=self.prefix,
            category=self.category,
            version=self.version,
            required_packages=self.required_packages,
            specs=selected,
        )

    def __iter__(self) -> Iterator[str]:
        """Yield canonical ID strings for all environments."""
        for spec in self.specs:
            yield self.get_name(spec.name)

    def __len__(self) -> int:
        """Return number of environments."""
        return len(self.specs)

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
    An ordered collection of `EnvSuite` instances.

    Combines multiple suites into a single iterable that yields canonical
    environment ID strings. Supports merging two `EnvSet` objects with `+`.

    Parameters
    ----------
    *suites : EnvSuite
        Variable number of environment suites to combine.
    """

    def __init__(self, *suites: EnvSuite) -> None:
        self._suites: List[EnvSuite] = list(suites)

    @property
    def n_envs(self) -> int:
        """Total number of environments across all suites."""
        return sum(s.n_envs for s in self._suites)

    @property
    def suites(self) -> List[EnvSuite]:
        """List of environment suites in this set."""
        return self._suites

    def all_names(self, version: str | None = None) -> List[str]:
        """
        Return canonical IDs for every environment across all suites.

        Parameters
        ----------
        version : str (optional)
            Override the default version suffix for all suites.

        Returns
        -------
        names : List[str]
        """
        names: List[str] = []
        for suite in self._suites:
            names.extend(suite.all_names(version))
        return names

    def env_categories(self) -> Dict[str, int]:
        """
        Return a mapping of category name → environment count.

        Returns
        -------
        categories : Dict[str, int]
        """
        counts: Dict[str, int] = {}
        for s in self._suites:
            counts[s.category] = counts.get(s.category, 0) + s.n_envs
        return counts

    def __iter__(self) -> Iterator[str]:
        """Yield canonical ID strings from all suites in order."""
        for suite in self._suites:
            yield from suite

    def __len__(self) -> int:
        """Total number of environments."""
        return self.n_envs

    def __add__(self, other: Self) -> Self:
        """Merge two EnvSets into a new one."""
        return type(self)(*self._suites, *other._suites)

    def verify_packages(self) -> None:
        """
        Verify that every suite has its required packages installed.

        Raises
        ------
        error : MissingPackageError
            If any suite has missing required packages.
        """
        missing: Dict[str, List[str]] = {}
        for suite in self._suites:
            status = suite.check()
            not_installed = [pkg for pkg, ok in status.items() if not ok]
            if not_installed:
                missing[suite.category] = not_installed

        if missing:
            lines = [f"  {cat}: {', '.join(pkgs)}" for cat, pkgs in missing.items()]
            raise MissingPackageError(
                "Missing required packages for environment suites:\n" + "\n".join(lines)
            )

    def __repr__(self) -> str:
        suite_info = ", ".join(
            f"{s.__class__.__name__}({s.n_envs})" for s in self._suites
        )
        return f"EnvSet({suite_info}, total={self.n_envs})"
