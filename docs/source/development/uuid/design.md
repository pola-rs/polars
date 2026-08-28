# Native UUID implementation design

This document describes the implementation choices behind Polars' native UUID data type. The public
API, examples, and user-facing behavior are documented in the
[UUID user guide](../../user-guide/concepts/uuids.md). This document instead concentrates on the
internal representation, standards compatibility, interchange boundaries, and measured behavior.

The implementation adds a native, feature-gated logical type across the Rust core, expression
engine, Python API, Arrow and Parquet interchange, text I/O, and SQL casting:

- Rust logical type: `DataType::Uuid`
- Python data type: `pl.UUID`
- physical representation: `UInt128`, 16 bytes per non-null value
- Arrow representation: canonical `arrow.uuid` over `FixedSizeBinary(16)`
- Parquet representation: `FIXED_LEN_BYTE_ARRAY(16)` with the UUID logical annotation
- Python scalar representation: `uuid.UUID`
- text representation: lowercase canonical `8-4-4-4-12` form

## Standards and compatibility basis

The implementation follows these primary definitions:

- [RFC 9562](https://www.rfc-editor.org/rfc/rfc9562.html) for UUID layout and versions, including
  versions 4 and 7.
- [Arrow canonical extension types](https://arrow.apache.org/docs/format/CanonicalExtensions.html)
  for `arrow.uuid` as 16-byte fixed-size binary storage.
- [Parquet UUID logical type](https://parquet.apache.org/docs/file-format/types/logicaltypes/#uuid)
  for big-endian 16-byte UUID storage.
- [PostgreSQL UUID input/output](https://www.postgresql.org/docs/current/datatype-uuid.html) for
  accepted text forms and canonical lowercase output.
- [Python's `uuid` module](https://docs.python.org/3/library/uuid.html) for Python scalar behavior.
- [DuckDB UUID functions and types](https://duckdb.org/docs/current/sql/functions/utility.html) for
  generation, introspection, and database interoperability.

Python's standard `uuid` module is the Python scalar and conversion contract. DuckDB's documented
UUID behavior and Python/Parquet interchange provide an independent database compatibility target.

## Core logical type

The `dtype-uuid` feature includes `dtype-u128` and is included by `dtype-full`. The logical type is
threaded through:

- `DataType`, `Field`, display/debug output, schema serialization, DSL schema hashes, and dtype
  conversion;
- `AnyValue::Uuid(u128)`, scalar serialization, hashing, equality, ordering, and formatting;
- `UuidType`, `UuidChunked`, typed downcasting, and full `SeriesTrait` dispatch; and
- logical-to-physical conversion and restoration.

Every 128-bit bit pattern is a valid UUID value. Storage does not restrict the UUID version or
variant; those properties are encoded within the value and can be inspected separately. Arithmetic
is rejected because the logical type is an identifier, even though its physical storage is an
integer.

### Byte order and ordering

The physical `u128` is numerically identical to the UUID's canonical big-endian bytes:

```text
UUID bytes -> u128::from_be_bytes(bytes)
u128       -> value.to_be_bytes()
```

Unsigned numeric ordering therefore agrees with canonical-byte lexicographic ordering and canonical
string ordering. It also makes UUIDv7 values naturally time ordered.

DuckDB flips the high bit in its internal representation because it stores UUIDs as signed 128-bit
integers and wants signed ordering to match UUID ordering. That is a DuckDB storage detail; using
the same transform at Polars' interchange boundaries would produce non-canonical Arrow and Parquet
bytes. Bidirectional Parquet tests confirm the standards-based representation.

### Parsing and formatting

The normal parser path uses the Rust `uuid` crate without allocation. A small fallback accepts the
additional PostgreSQL spellings: uppercase, no hyphens, braces, and hyphens after groups of four
hexadecimal digits. The `urn:uuid:` form accepted by Python and RFC tooling is also supported.

Output is always lowercase canonical text. UUID-to-string conversion uses a reusable fixed encode
buffer per chunk rather than allocating an intermediate Rust `String` per value. Strict casts reject
invalid values; non-strict casts replace them with null.

## Arrow, Parquet, and nested values

At Arrow boundaries UUID is represented as:

```text
Extension(name="arrow.uuid", storage=FixedSizeBinary(16))
```

Import validates both the canonical extension name and the exact 16-byte storage type. Export
converts physical integers to big-endian bytes. List fields retain the UUID extension rather than
degrading to binary or string.

Parquet writing emits the standard UUID logical annotation on a 16-byte fixed-length byte array.
Reading reconstructs the canonical Arrow extension and then the Polars logical dtype.

Parquet statistics need explicit handling: statistics arrays retain the extension dtype while
dispatching through fixed-size-binary storage. This is necessary for lazy UUID predicates during
statistics evaluation and is covered by a lazy scan predicate test.

## Python, expressions, I/O, and SQL

The Python boundary includes native dtype conversion and inference from `uuid.UUID`, construction
from UUID objects, strings, and 16-byte values, optional non-strict integer coercion, scalar/list
export as `uuid.UUID`, UUID scalar comparison broadcasting, UUID expression/series namespaces,
UUIDv4/v7 generation, and typed lazy-plan visitor nodes.

The implementation provides:

- `version()` extraction for every UUID version;
- strict or non-strict UUIDv7 timestamp extraction;
- UUIDv4 generation that batch-fills each output column from `rand`'s cryptographically secure
  thread-local generator and then patches the RFC 9562 version and variant bits in place;
- process-monotonic UUIDv7 generation using `now_v7()` and an ascending sorted flag; and
- non-deterministic expression classification so optimizer rewrites do not duplicate, cache, or
  factor generated values incorrectly.

The public generator count is explicit (`pl.uuid4(n)` and `pl.uuid7(n)`). Internally, the Rust
namespace can generate one UUID per input row. Explicit cardinality avoids a standalone Python
expression whose length silently changes with projection context.

Generator expressions are non-deterministic. Reusing a lazy query containing a generator in multiple
branches evaluates it independently in each branch. Call `collect` first or add an explicit
`LazyFrame.cache` when generated identifiers must be shared across branches.

I/O and SQL behavior is as follows:

- IPC preserves `arrow.uuid` metadata.
- Parquet preserves the standard UUID logical type and interoperates with DuckDB.
- CSV and JSON write canonical UUID strings.
- CSV schema overrides parse through strict UUID casts.
- JSON and NDJSON support direct UUID parsing; NDJSON also supports configurable error/null behavior
  and nested UUID lists. Both use the shared parser, including the PostgreSQL hyphen-grouping
  fallback.
- Iceberg UUID fields map to `pl.UUID` for native scans, sinks, and field defaults.
- SQL `CAST(... AS UUID)` produces native UUID values when `dtype-uuid` is enabled; builds without
  that feature retain the existing string fallback.

## Compatibility results

| Boundary            | Result                                                                                                      |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| Python `uuid.UUID`  | Native inference, construction, scalar comparisons, and scalar/list return values pass                      |
| PostgreSQL text     | Canonical, uppercase, compact, braced, and flexible four-digit grouping pass; output is canonical lowercase |
| Arrow               | `arrow.uuid` + `FixedSizeBinary(16)` scalar and nested round trips pass                                     |
| Parquet             | Logical type, nulls, statistics, eager/lazy reads, and predicate filtering pass                             |
| DuckDB Python 1.5.5 | Reads Polars Parquet as `UUID`; Polars reads DuckDB UUID Parquet as `pl.UUID`                               |
| Iceberg             | UUID schema fields, native scans/sinks, and field defaults round trip as `pl.UUID`                          |
| IPC                 | Canonical extension round trip passes                                                                       |
| CSV/JSON/NDJSON     | Canonical text writing and schema-directed reading pass                                                     |
| Polars SQL          | Native UUID cast passes                                                                                     |

The DuckDB check covers both directions with v4, v7, and null values. It can be reproduced with
[`duckdb_interop.py`](duckdb_interop.py).

## Benchmarks

The benchmark used one million UUIDv4 values, seven measured repetitions after two warmups, Python
3.12.7, ARM64 macOS 26.6.2, and Polars 2.0.0-rc.1. The Rust extension was built in release mode with
`opt-level=3` and thin LTO. These are directional single-machine microbenchmarks, not universal
performance claims.

### Storage

| Representation |            Memory | Uncompressed Parquet |
| -------------- | ----------------: | -------------------: |
| Native UUID    |  16,000,000 bytes |     16,004,855 bytes |
| String UUID    |  36,000,000 bytes |     40,012,151 bytes |
| Improvement    | **2.25× smaller** |    **2.50× smaller** |

The memory figures exclude a null bitmap because the generated benchmark column has no nulls.

### Median wall time

| Operation, 1M rows         | Native UUID | String UUID | Native speedup |
| -------------------------- | ----------: | ----------: | -------------: |
| Sort                       |    6.094 ms |   33.537 ms |      **5.50×** |
| `n_unique`                 |    6.552 ms |   14.924 ms |      **2.28×** |
| Equality filter            |    0.390 ms |    0.453 ms |      **1.16×** |
| Group by                   |    6.902 ms |   12.880 ms |      **1.87×** |
| Uncompressed Parquet write |    4.086 ms |    9.220 ms |      **2.26×** |

The group-by data contains 100,000 unique values repeated ten times and shuffled. Native and string
columns always contain the same UUIDs.

### Conversion and generation

| Operation, 1M rows |     Median | Approximate throughput |
| ------------------ | ---------: | ---------------------: |
| String to UUID     |  11.291 ms |         88.6 million/s |
| UUID to string     |  13.346 ms |         74.9 million/s |
| Generate UUIDv4    |   8.630 ms |        115.9 million/s |
| Generate UUIDv7    | 671.942 ms |         1.49 million/s |

The batched UUIDv4 implementation is approximately 68× faster than the earlier per-value generator
measurement. UUIDv7 remains bounded by its per-value monotonic context.

Run [`benchmark.py`](benchmark.py) to reproduce the benchmark. The recorded raw output and complete
min/median/max timings are in [`benchmark_results.json`](benchmark_results.json).

## Deliberate boundaries and extension points

The implementation is complete for its documented native UUID surface. The following adjacent
capabilities are intentionally left as later extensions:

- **UUIDv5 generation.** Storage accepts UUIDv5 and `version()` reports it, but deterministic
  namespace/name generation is not yet exposed. UUIDv4 and UUIDv7 are the initial generators.
- **PostgreSQL database writes.** PostgreSQL-compatible text parsing and database-read inference are
  covered, but `DataFrame.write_database` does not yet provide a UUID-specific SQLAlchemy/ADBC bind
  adapter or an integration test for native PostgreSQL UUID columns.
- **Avro logical UUID.** Avro annotates a string rather than 16-byte storage. Supporting it requires
  format-specific string transcoding in the Avro serializer/deserializer.
- **Generic extension and fixed-size-binary support.** Canonical `arrow.uuid` is recognized
  directly; this change does not implement arbitrary Arrow extension types, arbitrary field metadata
  preservation, or a general Polars fixed-size-binary dtype.
- **Higher-level UUID features.** UUID placeholders in partition filenames and UUID-based column
  lineage are independent features and are not part of the data type.

The current names (`DataType::Uuid`, `pl.UUID`), PostgreSQL-compatible parser, explicit generator
cardinality, and `dtype-uuid` feature gate are intentional API choices. They can evolve without
changing the 16-byte logical representation or canonical interchange contract.

## Reproducing the review evidence

```shell
cd py-polars
make test

python ../docs/source/development/uuid/benchmark.py
python ../docs/source/development/uuid/duckdb_interop.py
```

CodSpeed-compatible benchmark cases live in `py-polars/tests/benchmark/test_uuid.py`.
