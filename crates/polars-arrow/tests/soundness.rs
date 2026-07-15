use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::*;
use polars_arrow::offset::*;

// ===================================================================
// Category A: Integer overflow in slice()/sliced() bounds check
// ===================================================================

#[test]
#[should_panic]
fn fixed_size_binary_array_slice_overflow() {
    let dtype = ArrowDataType::FixedSizeBinary(4);
    let mut arr = FixedSizeBinaryArray::new(dtype, vec![0u8; 16].into(), None);
    arr.slice(usize::MAX - 1, 6);
}

#[test]
#[should_panic]
fn fixed_size_list_array_slice_overflow() {
    let child = PrimitiveArray::<i32>::from_vec(vec![1, 2, 3, 4]).boxed();
    let dtype = FixedSizeListArray::default_datatype(child.dtype().clone(), 2);
    let mut arr = FixedSizeListArray::new(dtype, 2, child, None);
    arr.slice(usize::MAX, 3);
}

#[test]
#[should_panic]
fn list_array_slice_overflow() {
    let child = PrimitiveArray::<i32>::from_vec(vec![10, 20]).boxed();
    let offsets = OffsetsBuffer::<i32>::try_from(vec![0, 1, 2]).unwrap();
    let dtype = ListArray::<i32>::default_datatype(ArrowDataType::Int32);
    let mut arr = ListArray::<i32>::new(dtype, offsets, child, None);
    arr.slice(usize::MAX - 1, 3);
}

#[test]
#[should_panic]
fn map_array_slice_overflow() {
    let entries = Field::new(
        "entries".into(),
        ArrowDataType::Struct(vec![
            Field::new("key".into(), ArrowDataType::Int32, false),
            Field::new("value".into(), ArrowDataType::Int32, true),
        ]),
        false,
    );
    let dtype = ArrowDataType::Map(Box::new(entries), false);
    let mut arr = MapArray::new_null(dtype, 1);
    arr.slice(usize::MAX, 2);
}

#[test]
#[should_panic]
fn struct_array_slice_overflow() {
    let child = PrimitiveArray::<i32>::from_vec(vec![10, 20, 30, 40]).boxed();
    let dtype = ArrowDataType::Struct(vec![Field::new("x".into(), ArrowDataType::Int32, false)]);
    let mut arr = StructArray::new(dtype, 4, vec![child], None);
    arr.slice(usize::MAX, 5);
}

#[test]
#[should_panic]
fn union_array_slice_overflow() {
    let dtype = ArrowDataType::Union(Box::new(UnionType {
        fields: vec![Field::new("a".into(), ArrowDataType::Int32, true)],
        ids: None,
        mode: UnionMode::Sparse,
    }));
    let child = PrimitiveArray::<i32>::from_vec(vec![111]).boxed();
    let types = vec![0i8].into();
    let mut arr = UnionArray::new(dtype, types, vec![child], None);
    arr.slice(usize::MAX, 2);
}

#[test]
#[should_panic]
fn utf8_array_slice_overflow() {
    let mut arr = Utf8Array::<i32>::from_slice(["A", "BB", "CCC"]);
    arr.slice(usize::MAX, 4);
}

#[test]
#[should_panic]
fn bitmap_slice_overflow() {
    let mut bm = Bitmap::from([true, false, true, false]);
    bm.slice(usize::MAX, 5);
}

#[test]
#[should_panic]
fn primitive_array_slice_overflow() {
    let mut arr = PrimitiveArray::<i32>::from_vec(vec![1, 2, 3]);
    arr.slice(1, usize::MAX);
}

#[test]
#[should_panic]
fn binary_array_slice_overflow() {
    let mut arr = BinaryArray::<i32>::from_slice([b"A" as &[u8], b"BB", b"CCC"]);
    arr.slice(usize::MAX, 4);
}

#[test]
#[should_panic]
fn boolean_array_slice_overflow() {
    let mut arr = BooleanArray::from_slice([true, false, true, false]);
    arr.slice(usize::MAX, 5);
}

#[test]
#[should_panic]
fn null_array_slice_overflow() {
    let mut arr = NullArray::new(ArrowDataType::Null, 3);
    arr.slice(usize::MAX, 4);
}

#[test]
#[should_panic]
fn bitmap_sliced_overflow() {
    let bm = Bitmap::from([true, false, true, false]);
    let _ = bm.sliced(usize::MAX, 5);
}

// ===================================================================
// Category B: fold sum overflow in extend_from_lengths
// ===================================================================

#[test]
#[should_panic]
fn binview_extend_from_lengths_overflow() {
    let mut arr = MutableBinaryViewArray::<[u8]>::new();
    let buffer = vec![0x41u8; 26];
    let lengths = vec![usize::MAX - 5, 32];
    arr.extend_from_lengths(&buffer, lengths.into_iter());
}

// ===================================================================
// Category C: buffer_idx u32 overflow in View::new_with_buffers
// ===================================================================

#[test]
#[should_panic]
fn view_new_with_buffers_overflow() {
    let mut raw = vec![vec![0u8; 16], Vec::with_capacity(20)];
    raw[1].extend_from_slice(&[b'B'; 20]);
    let _v = View::new_with_buffers(b"abcdefghijklmn", u32::MAX, &mut raw);
}
