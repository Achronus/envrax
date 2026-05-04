import os
import pathlib

import jax

DEFAULT_CACHE_DIR = pathlib.Path(
    os.environ.get(
        "ENVRAX_CACHE_DIR",
        str(pathlib.Path.home() / ".cache" / "envrax" / "xla_cache"),
    )
)

_cache_configured = False


def setup_cache(cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR) -> None:
    """
    Configure a persistent XLA compilation cache.

    Automatically appends the active JAX backend (`cpu`, `gpu`, or
    `tpu`) as a sub-directory so kernels for different backends never share
    the same path.  Idempotent — safe to call multiple times; only the first
    call takes effect.

    Parameters
    ----------
    cache_dir : Path | str | None
        Base directory for compiled XLA kernels.  The actual cache is stored
        at `{cache_dir}/{backend}/`.  Defaults to the `ENVRAX_CACHE_DIR`
        environment variable, or `~/.cache/envrax/xla_cache` if unset.
        Pass `None` to disable caching.
    """
    global _cache_configured

    if cache_dir is None or _cache_configured:
        return

    backend = jax.default_backend()
    path = pathlib.Path(cache_dir) / backend
    path.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
    _cache_configured = True
