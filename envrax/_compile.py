import os
import pathlib

import jax

DEFAULT_CACHE_DIR = pathlib.Path(
    os.environ.get("ENVRAX_CACHE_DIR", str(pathlib.Path.cwd() / ".jax_cache"))
)

_cache_configured = False


def setup_cache(cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR) -> None:
    """
    Configure a persistent XLA compilation cache.

    Idempotent — safe to call multiple times; only the first call takes
    effect.

    Parameters
    ----------
    cache_dir : Path | str | None (optional)
        Directory for compiled XLA kernels.  Defaults to the
        `ENVRAX_CACHE_DIR` environment variable, or `<cwd>/.jax_cache`
        if unset.  Pass `None` to disable caching.
    """
    global _cache_configured

    if cache_dir is None or _cache_configured:
        return

    path = pathlib.Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(path))
    _cache_configured = True
