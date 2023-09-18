# Scalar API

Design choices:

### `Scalar` is trait object

There are three reasons:

* a scalar should have a small memory footprint, which an enum would not ensure given the different physical types available.
* forward-compatibility: a new entry on an `enum` is backward-incompatible
* do not expose implementation details to users (reduce the surface of the public API)

### `Scalar` MUST contain nullability information

This is to be aligned with the general notion of arrow's `Array`.

This API is a companion to the `Array`, and follows the same design as `Array`.
Specifically, a `Scalar` is a trait object that can be downcasted to concrete implementations.

Like `Array`, `Scalar` implements

* `data_type`, which is used to perform the correct downcast
* `is_valid`, to tell whether the scalar is null or not

### There is one implementation per arrows' physical type

* Reduces the number of `match` that users need to write
* Allows casting of logical types without changing the underlying physical type
