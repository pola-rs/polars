use polars_error::PolarsResult;
use crate::array::*;
use crate::compute::cast::binary_to::Parse;
use crate::compute::cast::CastOptions;
use crate::datatypes::ArrowDataType;
use crate::offset::Offset;
use crate::types::NativeType;

pub(super) fn view_to_binary<O: Offset>(array: &BinaryViewArray) -> BinaryArray<O> {
    let len: usize = Array::len(array);
    let mut mutable = MutableBinaryValuesArray::<O>::with_capacities(len, len * 12);
    for slice in array.values_iter() {
        mutable.push(slice)
    }
    let out: BinaryArray<O> = mutable.into();
    out.with_validity(array.validity().cloned())
}

pub(super) fn utf8view_to_utf8<O: Offset>(array: &Utf8ViewArray) -> Utf8Array<O> {
    let array = array.to_binview();
    let out = view_to_binary::<O>(&array);

    let dtype = Utf8Array::<O>::default_data_type();
    unsafe {
        Utf8Array::new_unchecked(
            dtype,
            out.offsets().clone(),
            out.values().clone(),
            out.validity().cloned(),
        )
    }
}
/// Casts a [`BinaryArray`] to a [`PrimitiveArray`], making any uncastable value a Null.
pub(super) fn binview_to_primitive<T>(
    from: &BinaryViewArray,
    to: &ArrowDataType,
) -> PrimitiveArray<T>
    where
        T: NativeType + Parse,
{
    let iter = from.iter().map(|x| x.and_then::<T, _>(|x| T::parse(x)));

    PrimitiveArray::<T>::from_trusted_len_iter(iter).to(to.clone())
}

pub(super) fn binview_to_primitive_dyn<T>(
    from: &dyn Array,
    to: &ArrowDataType,
    options: CastOptions,
) -> PolarsResult<Box<dyn Array>>
    where
        T: NativeType + Parse,
{
    let from = from.as_any().downcast_ref().unwrap();
    if options.partial {
        unimplemented!()
    } else {
        Ok(Box::new(binview_to_primitive::<T>(from, to)))
    }
}
