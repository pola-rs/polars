use arrow::array::{BinaryViewArray, FixedSizeBinaryArray, PrimitiveArray};
use arrow::datatypes::ArrowDataType;
use polars_buffer::Buffer;
use polars_core::prelude::{Column, DataType, LargeBinaryArray};
use polars_core::with_match_physical_integer_type;

pub type PartitionKey = polars_utils::small_bytes::SmallBytes;

pub enum PreComputedKeys {
    Binview(BinaryViewArray),
    Primitive(FixedSizeBinaryArray),
    RowEncoded(LargeBinaryArray),
}

impl PreComputedKeys {
    #[expect(unused)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Binview(_) => "Binview",
            Self::Primitive(_) => "Primitive",
            Self::RowEncoded(_) => "RowEncoded",
        }
    }

    pub fn opt_new_non_encoded(column: &Column) -> Option<Self> {
        Some(match column.dtype() {
            DataType::Binary => Self::Binview(
                column
                    .binary()
                    .unwrap()
                    .rechunk()
                    .downcast_as_array()
                    .clone(),
            ),
            DataType::String => Self::Binview(
                column
                    .str()
                    .unwrap()
                    .as_binary()
                    .rechunk()
                    .downcast_as_array()
                    .clone(),
            ),
            dt if dt.is_primitive() && dt.to_physical().is_integer() => {
                let c = column.to_physical_repr();

                let [arr] = c
                    .as_materialized_series()
                    .rechunk()
                    .into_chunks()
                    .try_into()
                    .unwrap();

                let (bytes, width): (Buffer<u8>, usize) = with_match_physical_integer_type!(dt, |$T| {
                    let arr: &PrimitiveArray<$T> = arr.as_any().downcast_ref().unwrap();
                    (arr.values().clone().try_transmute().unwrap(), std::mem::size_of::<$T>())
                });

                assert_eq!(width * arr.len(), bytes.len());

                let arr = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(width),
                    bytes,
                    arr.validity().cloned(),
                );

                PreComputedKeys::Primitive(arr)
            },
            _ => return None,
        })
    }

    #[inline]
    pub fn get_key(&self, idx: usize) -> PartitionKey {
        match self {
            Self::Binview(arr) => PartitionKey::from_opt_slice(arr.get(idx)),
            Self::Primitive(arr) => PartitionKey::from_opt_slice(arr.get(idx)),
            Self::RowEncoded(arr) => PartitionKey::from_slice(unsafe { arr.value_unchecked(idx) }),
        }
    }
}
