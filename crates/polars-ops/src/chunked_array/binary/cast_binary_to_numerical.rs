use arrow::array::{Array, BinaryViewArray, PrimitiveArray};
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

/// Trait for casting bytes to a primitive type
pub trait Cast {
    fn cast_le(val: &[u8]) -> Option<Self>
    where
        Self: Sized;
    fn cast_be(val: &[u8]) -> Option<Self>
    where
        Self: Sized;
}
macro_rules! impl_cast {
    ($primitive_type:ident) => {
        impl Cast for $primitive_type {
            fn cast_le(val: &[u8]) -> Option<Self> {
                Some($primitive_type::from_le_bytes(val.try_into().ok()?))
            }

            fn cast_be(val: &[u8]) -> Option<Self> {
                Some($primitive_type::from_be_bytes(val.try_into().ok()?))
            }
        }
    };
}

impl_cast!(i8);
impl_cast!(i16);
impl_cast!(i32);
impl_cast!(i64);
impl_cast!(i128);
impl_cast!(u8);
impl_cast!(u16);
impl_cast!(u32);
impl_cast!(u64);
impl_cast!(u128);
impl_cast!(f32);
impl_cast!(f64);

/// Casts a [`BinaryArray`] to a [`PrimitiveArray`], making any uncastable value a Null.
pub(super) fn cast_binview_to_primitive<T>(
    from: &BinaryViewArray,
    to: &ArrowDataType,
    is_little_endian: bool,
) -> PrimitiveArray<T>
where
    T: Cast + NativeType,
{
    let iter = from.iter().map(|x| {
        x.and_then::<T, _>(|x| {
            if is_little_endian {
                T::cast_le(x)
            } else {
                T::cast_be(x)
            }
        })
    });

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

/// Casts a [`BinaryArray`] to a [`PrimitiveArray`], making any uncastable value a Null.
pub(super) fn cast_binview_to_primitive_dyn<T>(
    from: &dyn Array,
    to: &ArrowDataType,
    is_little_endian: bool,
) -> PolarsResult<Box<dyn Array>>
where
    T: Cast + NativeType,
{
    let from = from.as_any().downcast_ref().unwrap();

    Ok(Box::new(cast_binview_to_primitive::<T>(
        from,
        to,
        is_little_endian,
    )))
}
