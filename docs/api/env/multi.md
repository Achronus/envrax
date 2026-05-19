# Multi-Environment

Classes for composing several *different* environments (and their vectorised variants) under a single unified handle.

`MultiEnv` holds a list of `JaxEnv` instances and dispatches via Python iteration. `MultiVecEnv` holds a dict of [`BatchedEnv`](batched.md) instances (keyed by env name, auto-derived from a list if you prefer) and dispatches inside one `jax.jit` boundary.

::: envrax.multi_env.MultiEnv

::: envrax.multi_vec_env.MultiVecEnv
