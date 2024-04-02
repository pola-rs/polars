# Test Files

These test files are taken from the C++ capnp testing code: [json](https://github.com/capnproto/capnproto/blob/v1.0.1.1/c%2B%2B/src/capnp/testdata/pretty.json) and associated [schema](https://github.com/capnproto/capnproto/blob/v1.0.1.1/c%2B%2B/src/capnp/test.capnp).
We have removed unsupported types (e.g. interface).

The union type is created by hand.

Convert json to binary using capnp cli tool:

```
$ capnp convert json:binary ../../schema/tests/test_all_types.capnp TestAllTypes < all_types.json > all_types.bin
$ capnp convert json:binary ../../schema/tests/test_all_types.capnp TestUnion < union0.json > union0.bin
$ capnp convert json:binary ../../schema/tests/test_all_types.capnp TestUnion < union1.json > union1.bin
$ cat union0.bin union1.bin > union.bin
```
