use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_core::prelude::{
    Column, DataType, PlBinaryArray, PlBinaryViewArray, PlFixedSizeBinaryArray,
    PlPrimitiveArray,
};
use polars_core::with_match_physical_integer_type;

pub type PartitionKey = polars_utils::small_bytes::SmallBytes;

pub enum PreComputedKeys {
    Binview(PlBinaryViewArray),
    Primitive(PlFixedSizeBinaryArray),
    RowEncoded(PlBinaryArray),
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

                let length = arr.len();
                let arr: PlFixedSizeBinaryArray = with_match_physical_integer_type!(dt, |$T| {
                    let arr: &PlPrimitiveArray<$T> = arr.as_any().downcast_ref().unwrap();
                    let width = std::mem::size_of::<$T>();

                    // A scalar chunk holds the one value every element covers, so the keys are
                    // scalar too: the bytes are laid out once rather than once per row.
                    match arr.scalar_value() {
                        Some(value) => {
                            let bytes = Buffer::from(vec![value.unwrap_or_default()]);
                            PlFixedSizeBinaryArray::new_broadcast(
                                bytes.try_transmute().unwrap(),
                                width,
                                length,
                                value.is_none().then(|| Bitmap::new_zeroed(1)),
                            )
                        },
                        None => {
                            let flat = arr.to_flat();
                            PlFixedSizeBinaryArray::new(
                                flat.values().clone().try_transmute().unwrap(),
                                width,
                                length,
                                flat.validity().cloned(),
                            )
                        },
                    }
                });

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
