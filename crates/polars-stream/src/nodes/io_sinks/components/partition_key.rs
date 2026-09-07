use polars_array::PlBitmap;
use polars_buffer::Buffer;
use polars_core::prelude::{
    Column, DataType, PlBinaryArray, PlBinaryViewArray, PlFixedSizeBinaryArray, PlPrimitiveArray,
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
                                // The mask covers every element, so it is built for `length`
                                // rather than for the single bit that backs it.
                                value.is_none().then(|| PlBitmap::new_scalar(false, length)),
                            )
                        },
                        None => {
                            let flat = arr.to_flat();
                            PlFixedSizeBinaryArray::new(
                                flat.values().clone().try_transmute().unwrap(),
                                width,
                                length,
                                flat.validity().cloned().map(PlBitmap::from_bitmap),
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

#[cfg(test)]
mod tests {
    use polars_core::prelude::*;
    use polars_core::scalar::Scalar;

    use super::{PartitionKey, PreComputedKeys};

    /// A column that repeats one value keeps the keys scalar, and the mask that says the repeated
    /// value is null covers every element rather than the single bit that backs it.
    #[test]
    fn a_repeated_null_key_covers_every_row() {
        let length = 5;

        for (scalar, expected) in [
            (Scalar::null(DataType::Int64), PartitionKey::NULL),
            (
                Scalar::from(7i64),
                PartitionKey::from_slice(&7i64.to_ne_bytes()),
            ),
        ] {
            let column = Column::new_scalar("k".into(), scalar, length);
            let keys = PreComputedKeys::opt_new_non_encoded(&column).unwrap();

            for i in 0..length {
                assert_eq!(keys.get_key(i), expected);
            }
        }
    }
}
