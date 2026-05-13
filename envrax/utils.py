from math import prod
from types import get_original_bases
from typing import TYPE_CHECKING, Type, TypeVar, get_args, get_origin

if TYPE_CHECKING:
    from envrax.spaces import Space


def flat_size(space: "Space", env_name: str, kind: str) -> int:
    """
    Return `prod(space.shape)`. Raises `TypeError` if `space` has no `.shape`.

    Parameters
    ----------
    space : Space
        Space-like with a `.shape` tuple.
    env_name : str
        Env name, for the error message.
    kind : str
        `"action"` or `"observation"`, for the error message.

    Returns
    -------
    size : int
        Total flat element count.
    """
    shape = getattr(space, "shape", None)

    if shape is None:
        raise TypeError(
            f"Cannot compute flat size for env {env_name!r}: "
            f"{kind} space {type(space).__name__} has no `.shape`. "
            "Only spaces exposing a `.shape` tuple are supported."
        )

    return int(prod(shape))


def resolve_generic_arg(
    cls: Type,
    generic_base: Type,
    position: int,
) -> Type:
    """
    Resolve a concrete type pinned at a given position of a parameterised base class.

    Walks `cls`'s original bases looking for one whose origin matches `generic_base`
    and returns the type argument at `position`.
    Raises if no concrete type is found.

    Useful for runtime introspection of `Generic[...]` subclasses where you need
    to know which concrete type was pinned to a `TypeVar`.

    Parameters
    ----------
    cls : Type
        The subclass to introspect
    generic_base : Type
        The parameterised base class whose `TypeVar` is being resolved
    position : int
        Zero-based index of the `TypeVar` within
        `generic_base`'s type parameters

    Returns
    -------
    resolved : Type
        The concrete class pinned at `position`

    Raises
    ------
    wrong_cls: TypeError
        If `cls` does not directly subscript `generic_base`, or the resolved
        argument at `position` is still an unbound `TypeVar`

    Examples
    --------
    >>> class Foo(Generic[T, U]): ...
    >>> class Bar(Foo[int, str]): ...
    >>> resolve_generic_arg(Bar, Foo, position=0)
    <class 'int'>
    >>> resolve_generic_arg(Bar, Foo, position=1)
    <class 'str'>
    """
    for base in get_original_bases(cls):
        if get_origin(base) is generic_base:
            args = get_args(base)

            if position < len(args) and not isinstance(args[position], TypeVar):
                return args[position]

    raise TypeError(
        f"'{cls.__name__}' does not pin a concrete type at position '{position}' "
        f"of '{generic_base.__name__}'. Subscribe with "
        f"'{generic_base.__name__}[..., <your type>]' to fix."
    )
