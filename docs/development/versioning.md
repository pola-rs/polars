# Versioning

## Version changes

Polars adheres to the [semantic versioning](https://semver.org/) specification.

As Polars has not released its `1.0.0` version yet, breaking releases lead to a minor version increase (e.g. from `0.18.15` to `0.19.0`), while all other releases increment the patch version (e.g. from `0.18.14` to `0.18.15`)

## Policy for breaking changes

Polars takes backwards compatibility seriously, but we are not afraid to change things if it leads to a better product.

!!! warning Rust users only

    The Rust API for Polars is currently not considered stable.
    Functionality can be changed or removed without warning.

### Philosophy

We don't always get it right on the first try.
We learn as we go along and get feedback from our users.
Sometimes, we're a little too eager to get out a new feature and didn't ponder all the possible implications.

If this happens, we correct our mistakes and introduce a breaking change.
Most of the time, this is no big deal.
Users get a deprecation warning, they do a quick search-and-replace in their code base, and that's that.

At times, we run into an issue requires more effort on our user's part to fix.
A change in the query engine can seriously impact the assumptions in a data pipeline.
We do not make such changes lightly, but we will make them if we believe it makes Polars better.

Freeing ourselves of past indiscretions is important to keep Polars moving forward.
We know it takes time and energy for our users to keep up with new releases but, in the end, it benefits everyone for Polars to be the best product possible.

### What qualifies as a breaking change

**A breaking change occurs when an existing component of the public API is changed or removed.**

A feature is part of the public API if it is documented in the [API reference](https://pola-rs.github.io/polars/py-polars/html/reference/).

Examples of breaking changes:

- A deprecated function or method is removed.
- The default value of a parameter is changed.
- The outcome of a query has changed due to changes to the query engine.

Examples of changes that are _not_ considered breaking:

- An undocumented function is removed.
- The module path of a public class is changed.
- An optional parameter is added to an existing method.

Bug fixes are not considered a breaking change, even though it may impact some users' [workflows](https://xkcd.com/1172/).

### Deprecation warnings

If we decide to introduce a breaking change, the existing behavior is deprecated _if possible_.
For example, if we choose to rename a function, the new function is added alongside the old function, and using the old function will result in a deprecation warning.

Not all changes can be deprecated nicely.
A change to the query engine may have effects across a large part of the API.
Such changes will not be warned for, but _will_ be included in the changelog and the migration guide.

### Deprecation period

As a rule, deprecated functionality is removed two breaking releases after the deprecation happens.
For example, a function deprecated in version `0.18.3` will be removed in version `0.20.0`.

This means that if your program does not raise any deprecation warnings, it should be mostly safe to upgrade to the next breaking release.
As breaking releases happen about once every three months, this allows three to six months to adjust to any pending breaking changes.

**In some cases, we may decide to adjust the deprecation period.**
If retaining the deprecated functionality blocks other improvements to Polars, we may shorten the deprecation period to a single breaking release. This will be mentioned in the warning message.
If the deprecation affects many users, we may extend the deprecation period.
