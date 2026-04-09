import contextlib
import os
import pathlib
import threading

import jax
from tqdm import tqdm

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


@contextlib.contextmanager
def _live_bar(bar: tqdm, interval: float = 0.1):
    """
    Refresh *bar* from a background thread while the body executes.

    tqdm only redraws when `update()` is called, so without this the elapsed
    timer appears frozen during a long blocking operation such as XLA
    compilation.

    Parameters
    ----------
    bar : tqdm
        The progress bar instance to refresh.
    interval : float (optional)
        Seconds between refreshes. Default is `0.1`.
    """
    stop = threading.Event()

    def _spin():
        while not stop.is_set():
            bar.refresh()
            stop.wait(interval)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
