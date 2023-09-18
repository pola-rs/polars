use crate::offset::Offset;
use crate::types::NativeType;

use super::*;

mod binary;
mod boolean;
mod dictionary;
mod fixed_size_binary;
mod fixed_size_list;
mod list;
mod map;
mod null;
mod primitive;
mod struct_;
mod union;
mod utf8;

impl PartialEq for dyn Array + '_ {
    fn eq(&self, that: &dyn Array) -> bool {
        equal(self, that)
    }
}

impl PartialEq<dyn Array> for std::sync::Arc<dyn Array + '_> {
    fn eq(&self, that: &dyn Array) -> bool {
        equal(&**self, that)
    }
}

impl PartialEq<dyn Array> for Box<dyn Array + '_> {
    fn eq(&self, that: &dyn Array) -> bool {
        equal(&**self, that)
    }
}

impl PartialEq<NullArray> for NullArray {
    fn eq(&self, other: &Self) -> bool {
        null::equal(self, other)
    }
}

impl PartialEq<&dyn Array> for NullArray {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl<T: NativeType> PartialEq<&dyn Array> for PrimitiveArray<T> {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl<T: NativeType> PartialEq<PrimitiveArray<T>> for &dyn Array {
    fn eq(&self, other: &PrimitiveArray<T>) -> bool {
        equal(*self, other)
    }
}

impl<T: NativeType> PartialEq<PrimitiveArray<T>> for PrimitiveArray<T> {
    fn eq(&self, other: &Self) -> bool {
        primitive::equal::<T>(self, other)
    }
}

impl PartialEq<BooleanArray> for BooleanArray {
    fn eq(&self, other: &Self) -> bool {
        equal(self, other)
    }
}

impl PartialEq<&dyn Array> for BooleanArray {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl<O: Offset> PartialEq<Utf8Array<O>> for Utf8Array<O> {
    fn eq(&self, other: &Self) -> bool {
        utf8::equal(self, other)
    }
}

impl<O: Offset> PartialEq<&dyn Array> for Utf8Array<O> {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl<O: Offset> PartialEq<Utf8Array<O>> for &dyn Array {
    fn eq(&self, other: &Utf8Array<O>) -> bool {
        equal(*self, other)
    }
}

impl<O: Offset> PartialEq<BinaryArray<O>> for BinaryArray<O> {
    fn eq(&self, other: &Self) -> bool {
        binary::equal(self, other)
    }
}

impl<O: Offset> PartialEq<&dyn Array> for BinaryArray<O> {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl<O: Offset> PartialEq<BinaryArray<O>> for &dyn Array {
    fn eq(&self, other: &BinaryArray<O>) -> bool {
        equal(*self, other)
    }
}

impl PartialEq<FixedSizeBinaryArray> for FixedSizeBinaryArray {
    fn eq(&self, other: &Self) -> bool {
        fixed_size_binary::equal(self, other)
    }
}

impl PartialEq<&dyn Array> for FixedSizeBinaryArray {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl<O: Offset> PartialEq<ListArray<O>> for ListArray<O> {
    fn eq(&self, other: &Self) -> bool {
        list::equal(self, other)
    }
}

impl<O: Offset> PartialEq<&dyn Array> for ListArray<O> {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl PartialEq<FixedSizeListArray> for FixedSizeListArray {
    fn eq(&self, other: &Self) -> bool {
        fixed_size_list::equal(self, other)
    }
}

impl PartialEq<&dyn Array> for FixedSizeListArray {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl PartialEq<StructArray> for StructArray {
    fn eq(&self, other: &Self) -> bool {
        struct_::equal(self, other)
    }
}

impl PartialEq<&dyn Array> for StructArray {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl<K: DictionaryKey> PartialEq<DictionaryArray<K>> for DictionaryArray<K> {
    fn eq(&self, other: &Self) -> bool {
        dictionary::equal(self, other)
    }
}

impl<K: DictionaryKey> PartialEq<&dyn Array> for DictionaryArray<K> {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl PartialEq<UnionArray> for UnionArray {
    fn eq(&self, other: &Self) -> bool {
        union::equal(self, other)
    }
}

impl PartialEq<&dyn Array> for UnionArray {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

impl PartialEq<MapArray> for MapArray {
    fn eq(&self, other: &Self) -> bool {
        map::equal(self, other)
    }
}

impl PartialEq<&dyn Array> for MapArray {
    fn eq(&self, other: &&dyn Array) -> bool {
        equal(self, *other)
    }
}

/// Logically compares two [`Array`]s.
/// Two arrays are logically equal if and only if:
/// * their data types are equal
/// * each of their items are equal
pub fn equal(lhs: &dyn Array, rhs: &dyn Array) -> bool {
    if lhs.data_type() != rhs.data_type() {
        return false;
    }

    use crate::datatypes::PhysicalType::*;
    match lhs.data_type().to_physical_type() {
        Null => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            null::equal(lhs, rhs)
        }
        Boolean => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            boolean::equal(lhs, rhs)
        }
        Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            primitive::equal::<$T>(lhs, rhs)
        }),
        Utf8 => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            utf8::equal::<i32>(lhs, rhs)
        }
        LargeUtf8 => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            utf8::equal::<i64>(lhs, rhs)
        }
        Binary => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            binary::equal::<i32>(lhs, rhs)
        }
        LargeBinary => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            binary::equal::<i64>(lhs, rhs)
        }
        List => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            list::equal::<i32>(lhs, rhs)
        }
        LargeList => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            list::equal::<i64>(lhs, rhs)
        }
        Struct => {
            let lhs = lhs.as_any().downcast_ref::<StructArray>().unwrap();
            let rhs = rhs.as_any().downcast_ref::<StructArray>().unwrap();
            struct_::equal(lhs, rhs)
        }
        Dictionary(key_type) => {
            match_integer_type!(key_type, |$T| {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                dictionary::equal::<$T>(lhs, rhs)
            })
        }
        FixedSizeBinary => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            fixed_size_binary::equal(lhs, rhs)
        }
        FixedSizeList => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            fixed_size_list::equal(lhs, rhs)
        }
        Union => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            union::equal(lhs, rhs)
        }
        Map => {
            let lhs = lhs.as_any().downcast_ref().unwrap();
            let rhs = rhs.as_any().downcast_ref().unwrap();
            map::equal(lhs, rhs)
        }
    }
}
