use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::{ArrowDataType, Field, IntegerType, UnionMode, UnionType};
use polars_arrow::ffi;
use polars_arrow::offset::OffsetsBuffer;
use polars_buffer::Buffer;
use polars_error::PolarsResult;

fn _test_round_trip(array: Box<dyn Array>, expected: Box<dyn Array>) -> PolarsResult<()> {
    let field = Field::new("a".into(), array.dtype().clone(), true);

    // export array and corresponding dtype
    let array_ffi = ffi::export_array_to_c(array);
    let schema_ffi = ffi::export_field_to_c(&field);

    // import references
    let result_field = unsafe { ffi::import_field_from_c(&schema_ffi)? };
    let result_array = unsafe { ffi::import_array_from_c(array_ffi, result_field.dtype.clone())? };

    assert_eq!(&result_array, &expected);
    assert_eq!(result_field, field);
    Ok(())
}

fn test_round_trip(expected: impl Array + Clone + 'static) -> PolarsResult<()> {
    let array: Box<dyn Array> = Box::new(expected.clone());
    let expected = Box::new(expected) as Box<dyn Array>;
    _test_round_trip(array.clone(), clone(expected.as_ref()))?;

    // sliced
    _test_round_trip(array.sliced(1, 2), expected.sliced(1, 2))
}

#[test]
fn bool_nullable() -> PolarsResult<()> {
    let data = BooleanArray::from(&[Some(true), None, Some(false), None]);
    test_round_trip(data)
}

#[test]
fn binview_nullable_inlined() -> PolarsResult<()> {
    let data = Utf8ViewArray::from_slice([Some("foo"), None, Some("barbar"), None]);
    test_round_trip(data)
}

#[test]
fn binview_nullable_buffered() -> PolarsResult<()> {
    let data = Utf8ViewArray::from_slice([
        Some("foobaroiwalksdfjoiei"),
        None,
        Some("barbar"),
        None,
        Some("aoisejiofjfoiewjjwfoiwejfo"),
    ]);
    test_round_trip(data)
}

/// Explicit `ids`: the C format string always carries type ids, so `None` would not round-trip.
fn union(mode: UnionMode) -> UnionArray {
    let fields = vec![
        Field::new("i".into(), ArrowDataType::Int32, true),
        Field::new("s".into(), ArrowDataType::LargeUtf8, true),
    ];
    let dtype = ArrowDataType::Union(Box::new(UnionType {
        fields,
        ids: Some(vec![0, 1]),
        mode,
    }));
    let children: Vec<Box<dyn Array>> = vec![
        Box::new(PrimitiveArray::<i32>::from_slice([1, 2, 3, 4])),
        Box::new(Utf8Array::<i64>::from_slice(["a", "bb", "ccc", "dddd"])),
    ];
    let offsets = matches!(mode, UnionMode::Dense).then(|| vec![0i32, 1, 2, 3].into());
    UnionArray::new(dtype, vec![0i8, 1, 0, 1].into(), children, offsets)
}

#[test]
fn union_sparse() -> PolarsResult<()> {
    test_round_trip(union(UnionMode::Sparse))
}

#[test]
fn union_dense() -> PolarsResult<()> {
    test_round_trip(union(UnionMode::Dense))
}

#[test]
fn primitive_nullable() -> PolarsResult<()> {
    let data = PrimitiveArray::<i32>::from([Some(1), None, Some(3), Some(4), None]);
    test_round_trip(data)
}

#[test]
fn null() -> PolarsResult<()> {
    test_round_trip(NullArray::new(ArrowDataType::Null, 4))
}

#[test]
fn utf8_nullable() -> PolarsResult<()> {
    let data = Utf8Array::<i64>::from([Some("a"), None, Some("bb"), None, Some("ccc")]);
    test_round_trip(data)
}

#[test]
fn utf8_small_nullable() -> PolarsResult<()> {
    let data = Utf8Array::<i32>::from([Some("a"), None, Some("bb"), None, Some("ccc")]);
    test_round_trip(data)
}

#[test]
fn binary_nullable() -> PolarsResult<()> {
    let data = BinaryArray::<i64>::from([Some(b"a".as_ref()), None, Some(b"bb".as_ref()), None]);
    test_round_trip(data)
}

#[test]
fn binary_small_nullable() -> PolarsResult<()> {
    let data = BinaryArray::<i32>::from([Some(b"a".as_ref()), None, Some(b"bb".as_ref()), None]);
    test_round_trip(data)
}

#[test]
fn fixed_size_binary_nullable() -> PolarsResult<()> {
    let data = FixedSizeBinaryArray::new(
        ArrowDataType::FixedSizeBinary(2),
        Buffer::from(b"aabbccdd".to_vec()),
        Some(Bitmap::from([true, false, true, true])),
    );
    test_round_trip(data)
}

fn int_values() -> Box<dyn Array> {
    Box::new(PrimitiveArray::<i32>::from_slice([1, 2, 3, 4, 5, 6, 7, 8]))
}

#[test]
fn large_list_nullable() -> PolarsResult<()> {
    let data = ListArray::<i64>::new(
        ArrowDataType::LargeList(Box::new(Field::new(
            "item".into(),
            ArrowDataType::Int32,
            true,
        ))),
        OffsetsBuffer::<i64>::try_from(vec![0i64, 2, 4, 6, 8])?,
        int_values(),
        Some(Bitmap::from([true, false, true, true])),
    );
    test_round_trip(data)
}

#[test]
fn list_nullable() -> PolarsResult<()> {
    let data = ListArray::<i32>::new(
        ArrowDataType::List(Box::new(Field::new(
            "item".into(),
            ArrowDataType::Int32,
            true,
        ))),
        OffsetsBuffer::<i32>::try_from(vec![0i32, 2, 4, 6, 8])?,
        int_values(),
        Some(Bitmap::from([true, false, true, true])),
    );
    test_round_trip(data)
}

#[test]
fn fixed_size_list_nullable() -> PolarsResult<()> {
    let data = FixedSizeListArray::new(
        ArrowDataType::FixedSizeList(
            Box::new(Field::new("item".into(), ArrowDataType::Int32, true)),
            2,
        ),
        4,
        int_values(),
        Some(Bitmap::from([true, false, true, true])),
    );
    test_round_trip(data)
}

fn struct_fields() -> Vec<Field> {
    vec![
        Field::new("a".into(), ArrowDataType::Int32, true),
        Field::new("b".into(), ArrowDataType::LargeUtf8, true),
    ]
}

#[test]
fn struct_nullable() -> PolarsResult<()> {
    let data = StructArray::new(
        ArrowDataType::Struct(struct_fields()),
        4,
        vec![
            Box::new(PrimitiveArray::<i32>::from([
                Some(1),
                None,
                Some(3),
                Some(4),
            ])) as Box<dyn Array>,
            Box::new(Utf8Array::<i64>::from_slice(["a", "bb", "ccc", "dddd"])) as Box<dyn Array>,
        ],
        Some(Bitmap::from([true, false, true, true])),
    );
    test_round_trip(data)
}

/// Children that carry a slice offset of their own, independent of the parent's. Exporting a
/// buffer at an offset the consumer applies a second time reads out of bounds here.
#[test]
fn struct_of_sliced_children() -> PolarsResult<()> {
    let ints = PrimitiveArray::<i32>::from_slice([0, 1, 2, 3, 4, 5, 6, 7]).sliced(2, 4);
    let strings = Utf8Array::<i64>::from_slice(["z0", "z1", "a", "bb", "ccc", "dddd"]).sliced(2, 4);
    let data = StructArray::new(
        ArrowDataType::Struct(struct_fields()),
        4,
        vec![
            Box::new(ints) as Box<dyn Array>,
            Box::new(strings) as Box<dyn Array>,
        ],
        Some(Bitmap::from([true, false, true, true])),
    );
    test_round_trip(data)
}

#[test]
fn map_nullable() -> PolarsResult<()> {
    let entries = StructArray::new(
        ArrowDataType::Struct(vec![
            Field::new("key".into(), ArrowDataType::LargeUtf8, false),
            Field::new("value".into(), ArrowDataType::Int32, true),
        ]),
        4,
        vec![
            Box::new(Utf8Array::<i64>::from_slice(["k0", "k1", "k2", "k3"])) as Box<dyn Array>,
            Box::new(PrimitiveArray::<i32>::from([
                Some(0),
                None,
                Some(2),
                Some(3),
            ])) as Box<dyn Array>,
        ],
        None,
    );
    let data = MapArray::new(
        ArrowDataType::Map(
            Box::new(Field::new("entries".into(), entries.dtype().clone(), false)),
            false,
        ),
        OffsetsBuffer::<i32>::try_from(vec![0i32, 1, 2, 3, 4])?,
        Box::new(entries),
        Some(Bitmap::from([true, false, true, true])),
    );
    test_round_trip(data)
}

#[test]
fn dictionary_nullable() -> PolarsResult<()> {
    let data = DictionaryArray::<i32>::try_new(
        ArrowDataType::Dictionary(
            IntegerType::Int32,
            Box::new(ArrowDataType::LargeUtf8),
            false,
        ),
        PrimitiveArray::<i32>::from([Some(0), None, Some(1), Some(0)]),
        Box::new(Utf8Array::<i64>::from_slice(["x", "yy"])),
    )?;
    test_round_trip(data)
}
