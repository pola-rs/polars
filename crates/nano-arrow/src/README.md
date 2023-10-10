# Crate's design

This document describes the design of this module, and thus the overall crate.
Each module MAY have its own design document, that concerns specifics of that module, and if yes,
it MUST be on each module's `README.md`.

## Equality

Array equality is not defined in the Arrow specification. This crate follows the intent of the specification, but there is no guarantee that this no verification that this equals e.g. C++'s definition.

There is a single source of truth about whether two arrays are equal, and that is via their
equality operators, defined on the module [`array/equal`](array/equal/mod.rs).

Implementation MUST use these operators for asserting equality, so that all testing follows the same definition of array equality.

## Error handling

- Errors from an external dependency MUST be encapsulated on `External`.
- Errors from IO MUST be encapsulated on `Io`.
- This crate MAY return `NotYetImplemented` when the functionality does not exist, or it MAY panic with `unimplemented!`.

## Logical and physical types

There is a strict separation between physical and logical types:

- physical types MUST be implemented via generics
- logical types MUST be implemented via variables (whose value is e.g. an `enum`)
- logical types MUST be declared and implemented on the `datatypes` module

## Source of undefined behavior

There is one, and only one, acceptable source of undefined behavior: FFI. It is impossible to prove that data passed via pointers are safe for consumption (only a promise from the specification).
