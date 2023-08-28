# Data types

`Polars` is entirely based on `Arrow` data types and backed by `Arrow` memory arrays. This makes data processing
cache-efficient and well-supported for Inter Process Communication. Most data types follow the exact implementation
from `Arrow`, with the exception of `Utf8` (this is actually `LargeUtf8`), `Categorical`, and `Object` (support is limited). The data types are:

|Group| Type | Details |
|-----|-------|-------------|
|Numeric| `Int8`| 8-bit signed integer.|
|| `Int16`| 16-bit signed integer.|
|| `Int32`| 32-bit signed integer.|
|| `Int64`| 64-bit signed integer.|
|| `UInt8`| 8-bit unsigned integer.|
|| `UInt16`| 16-bit unsigned integer.|
|| `UInt32`| 32-bit unsigned integer.|
|| `UInt64`| 64-bit unsigned integer.|
|| `Float32`| 32-bit floating point.|
|| `Float64`| 64-bit floating point.|
|Nested| `Struct`| A struct array is represented as a `Vec<Series>` and is useful to pack multiple/heterogenous values in a single column.|
|| `List`| A list array contains a child array containing the list values and an offset array. (this is actually `Arrow` `LargeList` internally).|
|Temporal| `Date`| Date representation, internally represented as days since UNIX epoch encoded by a 32-bit signed integer.|
|| `Datetime`| Datetime representation, internally represented as microseconds since UNIX epoch encoded by a 64-bit signed integer.|
|| `Duration`| A timedelta type, internally represented as microseconds. Created when subtracting `Date/Datetime`.|
|| `Time`| Time representation, internally represented as nanoseconds since midnight.|
|Other| `Boolean`| Boolean type effectively bit packed.|
|| `Utf8`| String data (this is actually `Arrow` `LargeUtf8` internally).|
|| `Binary`| Store data as bytes.|
|| `Object`| A limited supported data type that can be any value.|
|| `Categorical` | 	A categorical encoding of a set of strings.|

To learn more about the internal representation of these data types, check the [`Arrow` columnar format](https://arrow.apache.org/docs/format/Columnar.html).