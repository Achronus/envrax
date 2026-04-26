# Environment Registry

Envrax ships with an easy way to create a shared environment registry that maps canonical environment *names* to their class, default config, and suite metadata. Once registered, any part of your code - training loops, evaluation scripts, configuration files - can refer to an environment by a string like `"BallEnv-v0"` instead of using the class directly.

As Envrax is the base API standard, the registry starts **empty**. Environments are contributed by your code (or by suite packages you install) via `register()` and `register_suite()`.

We'll focus on how this works and how to use these methods effectively throughout this tutorial. Without further ado, let's get into it! :rocket:

## Defining Environments

Before we explore to the `register()` methods, we first need to understand how environments are defined within the registry.

There are three main building blocks:

- `EnvSpec` - the individual specification for an environment.
- `EnvSuite` - a named, versioned collection of environment specs (`EnvSpec`) that belong to a single suite.
- `EnvSet` - an ordered collection of multiple `EnvSuite` instances.

We'll build our understanding of these first!

### EnvSpec

???+ api "API Docs"

    [`envrax.suite.EnvSpec`](../../api/registry/suite.md#envrax.suite.EnvSpec)

`EnvSpec` is a Python `dataclass` that holds the core information about an environment - it's name, class type, default config, and suite tag. You'll mostly build these inside an `EnvSuite.specs` list rather than constructing them directly.

Here's a simple example:

```python
from envrax import EnvSpec

from demo_envs.cartpole import CartpoleEnv, CartpoleConfig

spec = EnvSpec(
    name="cartpole",  # (1)
    env_class=CartpoleEnv, 
    default_config=CartpoleConfig(),
    # suite="MuJoCo",  # (2)
)
```

1. We recommend these to be lowercase
2. Populated automatically!

### EnvSuite

???+ api "API Docs"

    [`envrax.suite.EnvSuite`](../../api/registry/suite.md#envrax.suite.EnvSuite)

`EnvSuite` is made up of multiple `EnvSpecs` to create a suite of environments. This could be Atari games, Procgen games, DMLab environments, or any other custom *collection* of environments put under one banner.

They define a name (`prefix`) for the suite, a human-readable `category` label, a `version` for the suite, a list of the `required_packages`, and a list of the environment `specs` that belong to it.

This is used as a base class and **must** be inherited from to be able to use it.

Here's how to create one:

```python
from dataclasses import dataclass, field
from typing import List
from envrax import EnvSpec, EnvSuite

from demo_envs.cartpole import CartpoleEnv, CartpoleConfig
from demo_envs.ant import AntEnv, AntConfig

@dataclass
class DemoSuite(EnvSuite):
    prefix: str = "mjx"
    category: str = "MuJoCo Playground"
    version: str = "v0"
    required_packages: List[str] = field(
        default_factory=lambda: ["demo_envs"]
    )
    specs: List[EnvSpec] = field(
        default_factory=lambda: [
            EnvSpec(
                name="cartpole", 
                env_class=CartpoleEnv, 
                default_config=CartpoleConfig(),
            ),
            EnvSpec(
                name="ant",
                env_class=AntEnv,
                default_config=AntConfig(),
            ),
        ]
    )
```

Now, we get canonical IDs for each environment in the format: `mjx/cartpole-v0`.

That's not the only thing we can do though! `EnvSuite` comes with a few useful attributes and methods.

#### Attributes

There are two read-only attributes:

1. `envs` - lists the environment names of the suite
2. `n_envs` - provides the total number of environments in the suite

```python
suite = DemoSuite()

suite.envs    # ['cartpole', 'ant']
suite.n_envs  # 2
```

#### `get_name()`

Builds the canonical ID for one environments short name. This is the one that's used within the `register()` and [`make()`](make.md) methods (more on those in the next tutorial).

The format is simple: `<prefix>/<env_name>-<suite_version>`.

We can also pass in an optional `version` parameter to get a specific version of the environment, if you've shipped multiple versions:

```python
suite.get_name("cartpole")              # 'mjx/cartpole-v0'
suite.get_name("cartpole", version="v1")  # 'mjx/cartpole-v1'
```

#### `all_names()`

Returns the canonical IDs for every spec in one shot. This is convenient for when you need a flat list (e.g. for logging which envs were registered, or displaying the catalog in a UI).

Again, we can pass in an optional `version` parameter to get a specific version of the environments:

```python
suite.all_names()              # ['mjx/cartpole-v0', 'mjx/ant-v0']
suite.all_names(version="v1")  # ['mjx/cartpole-v1', 'mjx/ant-v1']
```

#### `check()` & `is_available()`

Need to know if your suite has the right packages installed? We've got you covered!

The `check()` and `is_available()` methods explore the `required_packages` and verify if they are importable. `check()` returns a per-package status mapping, while `is_available()` collapses that into a single boolean:

```python
suite.check()         # {'demo_envs': True}
suite.is_available()  # True
```

#### Iteration, length, slicing, and membership

`EnvSuite` also comes with a standard collection of magic methods so it can behave just like a Python sequence of canonical IDs:

```python
list(suite)  # ['mjx/cartpole-v0', 'mjx/ant-v0']
len(suite)  # 2
"cartpole" in suite  # True
```

You can also slice a subset of the suite to get only a handful of them. Be warned, this does create a new `EnvSuite` instance:

```python
subset = suite[:1]
subset.envs  # ['cartpole']
```

### EnvSet

???+ api "API Docs"

    [`envrax.suite.EnvSet`](../../api/registry/suite.md#envrax.suite.EnvSet)

`EnvSet` groups multiple `EnvSuite` instances into a single iterable. This can be useful when you want to compose several benchmarks into one heterogeneous catalog (e.g. Atari + MuJoCo).

Unlike `EnvSuite`, you don't subclass it. You just pass any number of suites in and the set holds them in the order you provided:

```python
from envrax import EnvSet

catalog = EnvSet(DemoSuite(), AnotherSuite())
```

It also comes with its own attributes and methods.

#### Attributes

There are two read-only attributes:

1. `suites` - the list of `EnvSuite` instances stored in the set
2. `n_envs` - the total number of environments across **all** suites

```python
catalog.suites   # [DemoSuite(...), AnotherSuite(...)]
catalog.n_envs   # 7
```

#### `all_names()`

This returns the canonical IDs for every environment across **every** suite, flattened into one list.

It supports the same optional `version` parameter as `EnvSuite.all_names()` for when you want to use a different suite version:

```python
catalog.all_names()              # ['mjx/cartpole-v0', 'mjx/ant-v0', 'other/foo-v0', ...]
catalog.all_names(version="v1")  # ['mjx/cartpole-v1', 'mjx/ant-v1', ...]
```

#### `env_categories()`

Returns a `category → count` mapping summarising how many environments each suite has. This is handy for printing a quick catalog summary or grouping envs into a UI:

```python
catalog.env_categories()  # {'MuJoCo Playground': 2, 'Other Suite': 5}
```

#### `verify_packages()`

Walks every suite and checks that its `required_packages` are importable.

Unlike `EnvSuite.is_available()`, this one **raises** `MissingPackageError` listing every missing package per suite. It's useful to use as a one-shot guardrail that you can use at the top of a script or custom package:

```python
# raises MissingPackageError if any packages are missing
catalog.verify_packages()
```

#### Iteration, length, and merging

`EnvSet` also implements the standard collection of magic methods so it behaves like a flat sequence of canonical IDs across all suites:

```python
list(catalog)  # ['mjx/cartpole-v0', 'mjx/ant-v0', 'other/foo-v0', ...]
len(catalog)   # 7
```

You can also merge two sets together with the `+` operator. This returns a brand-new `EnvSet` containing the suites from both sides in order:

```python
merged = catalog + EnvSet(ThirdSuite())
```

That covers our three building blocks, let's move onto the `register()` methods!

## Environment Registration

One of the biggest benefits of [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) is it's unified environment registry so that you can easily pick and choose which environments you want just by using its canonical name.

Now that we've seen how to create our own environments and suites, we can add them to an Envrax registry. There are two ways to do this:

1. `register()` - for individual environments
2. `register_suite()` - for a single suite of environments

### `register()`

???+ api "API Docs"

    [`envrax.registry.register`](../../api/registry/registry.md#envrax.registry.register)

Can be used to register one environment, under a single name, without a dedicated `EnvSuite`:

```python
import envrax
from envrax import EnvConfig

envrax.register("BallEnv-v0", BallEnv, BallConfig())
```

This has four positional arguments:

1. **Canonical name** - this is the unique name used within the registry. By convention, end it with `-v<N>` so different versions can coexist.
2. **Environment class** — the environment class type to register.
3. **Default config** — the environment config to use with it.
4. **Suite name** (optional) - a human-readable suite category tag.

An example with the `suite` keyword:

```python
envrax.register("BallEnv-v0", BallEnv, BallConfig(), suite="Ball Environments")
```

### `register_suite()`

???+ api "API Docs"

    [`envrax.registry.register_suite`](../../api/registry/registry.md#envrax.registry.register_suite)

Once your `EnvSuite` is defined, use `register_suite()` to add all its environments to the Envrax registry in one shot:

```python
from envrax import register_suite

register_suite(DemoSuite())
# Registers: "demo/cartpole-v0", "demo/ant-v0"
```

Want to override the default version? Pass in a new one with the `version` parameter:

```python
from envrax import register_suite

register_suite(DemoSuite(), version="v1")
# Registers: "demo/cartpole-v1", "demo/ant-v1"
```

## Utility Methods

???+ api "API Docs"

    - [`envrax.registry.registered_names`](../../api/registry/registry.md#envrax.registry.registered_names)
    - [`envrax.registry.get_spec`](../../api/registry/registry.md#envrax.registry.get_spec)

If you want to quickly verify that a `EnvSpec` or suite of environments is registered correctly, you can use one of the following:

- `registered_names()` - provides a list of canonical IDs stored in the Envrax registry
- `get_spec()` - accepts a single canonical ID and returns it `EnvSpec`, if it exists

```python
envrax.registered_names()
# ['BallEnv-v0', 'demo/ant-v0', 'demo/cartpole-v0']

envrax.get_spec("demo/cartpole-v0")
# EnvSpec(
#    name='demo/cartpole-v0', 
#    env_class=CartpoleEnv,
#    default_config=CartpoleConfig(...), 
#    suite='Demo Suite',
# )
```

## When To Register Environments

Environments should only ever be registered to the registry **once**. The single rule we recommend following:

> **One file owns registration** that is then imported before using your `envrax.make()` methods.

There are two flavours of this pattern depending on what you're shipping.

### Library / Environment Suite

Put your `register_suite()` call inside the package's `__init__.py` file. When a user imports the package, it immediately triggers registration without users having to think about it.

```python
# demo_envs/__init__.py
from envrax import register_suite

from demo_envs.suite import DemoSuite

register_suite(DemoSuite())
```

Now any user just needs to use `import demo_envs` once before calling an [`envrax.make()`](make.md) method and the environment suite becomes available immediately :muscle:.

### Application / Research Code

For projects with small sample of custom environments that don't need an `EnvSuite` (training scripts, research code, evaluation pipelines), create a dedicated `registry.py` (or `envs/__init__.py`) that imports your env classes and registers them in one place:

```python
# myproject/registry.py
import envrax

from myproject.envs.ball import BallEnv, BallConfig
from myproject.envs.cartpole import CartpoleEnv, CartpoleConfig

envrax.register("BallEnv-v0", BallEnv, BallConfig())
envrax.register("CartPole-v0", CartpoleEnv, CartpoleConfig())
```

Then every file entry point - `train.py`, `eval.py`, `plot.py` - can use the same one-liner to populate the registry:

```python
import myproject.registry
```

One symmetric entry point with minimal effort! :smile:

### What To Avoid

- **Registering inside functions that run repeatedly.** The registry raises `ValueError` on duplicate names, so this will break your training loops.
- **Registering at the bottom of each environment file/`EnvSuite`.** This works, but it spreads the catalog across `N` files, makes side-effect-free imports impossible, and is easy to break by renaming or moving a file.
- **Copy-pasting `register()` calls into every entry point.** This inevitably can cause environments to drift out of sync and you get a confusing "Unknown environment" error.

## Common Pitfalls

Here's some common "gotcha's" to be mindful of:

- **`ValueError: Unknown environment: 'BallEnv-v0'`** - you called a [`make()`](make.md) method before the environment(s) were registered. Check your import order.
- **`ValueError: Environment 'BallEnv-v0' is already registered`** - re-registering an existing environment. Use a different version (`-v1`) or restart the process.

## Recap

Phew! We've covered a lot here so let's recap:

**The three building blocks**

- `EnvSpec` - holds one environment's `name`, `env_class`, `default_config`, and `suite` tag. Mostly built inside an `EnvSuite.specs` list.
- `EnvSuite` - a versioned, prefixed collection of `EnvSpec`s. Subclass it, set `prefix`/`category`/`version`/`required_packages`/`specs`.
- `EnvSet` - groups multiple `EnvSuite` instances into one iterable. Useful when composing several benchmarks into a single heterogeneous catalog.

**Useful methods on suites and sets**

- `EnvSuite`: `envs`, `n_envs`, `get_name()`, `all_names()`, `check()`, `is_available()`, plus iteration / `len()` / slicing / `in` support.
- `EnvSet`: `suites`, `n_envs`, `all_names()`, `env_categories()`, `verify_packages()`, plus iteration / `len()` / `+` for merging.

**Registering**

- `register(name, cls, default_config, suite="")` - adds one env to the registry. Best for prototyping or single-env projects.
- `register_suite(suite)` - registers every spec in an `EnvSuite` under its canonical IDs in one call. The standard path for shipping benchmarks.

**Introspecting**

- `registered_names()` - provides a sorted list of every registered canonical ID.
- `get_spec(name)` - returns the full `EnvSpec` for one environment, including its default config and suite tag.

**The golden rule**

One file owns registration. For library code, that's the package's `__init__.py`. For application code, that's a dedicated `registry.py` imported once at startup. Don't scatter `register()` calls across env files or entry points.

## Next Steps

Excellent job! You've got envs registered, now let's actually construct them. In the next tutorial we'll explore the `make` factory methods that look up registered names and hand back a fully-wired environment.

<div class="grid cards" markdown>

-   :material-cog-play-outline:{ .lg .middle } __Make Methods__

    ---

    Learn how to use Envrax's `make` factory methods.

    [:octicons-arrow-right-24: Continue to Tutorial 8](make.md)

</div>
