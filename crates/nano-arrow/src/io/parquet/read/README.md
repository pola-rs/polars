## Observations

### LSB equivalence between definition levels and bitmaps

When the maximum repetition level is 0 and the maximum definition level is 1,
the RLE-encoded definition levels correspond exactly to Arrow's bitmap and can be
memcopied without further transformations.

## Nested parquet groups are deserialized recursively

Reading a parquet nested field is done by reading each primitive
column sequentially, and build the nested struct recursively.

Rows of nested parquet groups are encoded in the repetition and definition levels.
In arrow, they correspond to:
* list's offsets and validity
* struct's validity

The implementation in this module leverages this observation:

Nested parquet fields are initially recursed over to gather
whether the type is a Struct or List, and whether it is required or optional, which we store
in `nested_info: Vec<Box<dyn Nested>>`. `Nested` is a trait object that receives definition
and repetition levels depending on the type and nullability of the nested item.
We process the definition and repetition levels into `nested_info`.

When we finish a field, we recursively pop from `nested_info` as we build
the `StructArray` or `ListArray`.

With this approach, the only difference vs flat is:
1. we do not leverage the bitmap optimization, and instead need to deserialize the repetition
   and definition levels to `i32`.
2. we deserialize definition levels twice, once to extend the values/nullability and
   one to extend `nested_info`.
