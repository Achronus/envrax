# API Reference

Welcome to the API documentation! Here you'll find everything you need to know about Envrax's classes, methods and functionality.

For flexibility, we've organised the reference by category, each covering a distinct part of the API surface.

Here's a quick overview:

- **Environments** - the core building blocks for defining and running RL environments. Here you'll find the [base classes](env/base.md) for building your own environments, how to [vectorise](env/vec.md) them and the classes used for managing [multiple](env/multi.md) at once!
- **[Spaces](spaces.md)** - the specifications/contracts for describing observation and action domains, including the well known `Discrete` and `Box` spaces from the [Gymnasium [:material-arrow-right-bottom:]](https://gymnasium.farama.org/) API.
- **Wrappers** - composable transformations that modify environment behaviour. Looking to build your own? Explore the [base classes](wrappers/base.md) and their [type variables](wrappers/types.md)! Otherwise, check out the existing stateless [pass-through](wrappers/passthrough.md) wrappers and [stateful](wrappers/stateful.md) variants that Envrax has to offer!
- **Environment Registry** - methods for publishing and looking up environments by ID. One [registry](registry/registry.md), unlimited potential. Use the [suite types](registry/suite.md) here to easily group related environments together for greater navigation across different suites!
- **[Make Methods](make.md)** - canonical factory functions for instantiating environments from registered IDs.
- **[Error Handling](error.md)** - custom exceptions raised by Envrax.

## Unsure Where to Start?

<div class="grid cards" markdown>

-   :material-creation-outline:{ .lg .middle } __Tutorials__

    ---

    Learn how to use Envrax, your way!

    [:octicons-arrow-right-24: Start learning](../tutorials/index.md)

</div>
