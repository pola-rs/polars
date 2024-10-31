# About

Polars releases an upgrade guide alongside each breaking release.
This guide is intended to help you upgrade from an older Polars version to the new version.

Each guide contains all breaking changes that were not previously deprecated, as well as any significant new deprecations.

A full list of all changes is available in the [changelog](../changelog.md).

!!! tip

    It can be useful to upgrade to the latest non-breaking version before upgrading to a new breaking version.
    This way, you can run your code and address any deprecation warnings.
    The upgrade to the new breaking version should then go much more smoothly!

!!! tip

    One of our maintainers has created a tool for automatically upgrading your Polars code to a later version.
    It's based on the well-known pyupgrade tool.
    Try out [polars-upgrade](https://github.com/MarcoGorelli/polars-upgrade) and let us know what you think!

!!! rust "Note"

    There are no upgrade guides yet for Rust releases.
    These will be added once the rate of breaking changes to the Rust API slows down and a [deprecation policy](../../development/versioning.md#deprecation-period) is added.
