use arrow::array::StructArray;
use arrow::datatypes::{DTYPE_CATEGORICAL, DTYPE_ENUM_VALUES, IntegerType};
use polars_compute::cast::CastOptionsImpl;

use self::categorical::CategoricalDecoder;
use self::nested::deserialize::utils::freeze_validity;
use self::nested_utils::NestedContent;
use self::utils::PageDecoder;
use super::*;
use crate::parquet::error::ParquetResult;

pub fn columns_to_iter_recursive(
    mut columns: Vec<BasicDecompressor>,
    mut types: Vec<&PrimitiveType>,
    field: Field,
    mut init: Vec<InitNested>,
    filter: Option<Filter>,
) -> ParquetResult<(NestedState, Box<dyn Array>, Bitmap)> {
    if !field.dtype().is_nested() {
        let pages = columns.pop().unwrap();
        init.push(InitNested::Primitive(field.is_nullable));
        let type_ = types.pop().unwrap();
        let (nested, arr, pdm) = page_iter_to_array(pages, type_, field, filter, Some(init))?;
        Ok((nested.unwrap(), arr, pdm))
    } else {
        match field.dtype() {
            ArrowDataType::List(inner) | ArrowDataType::LargeList(inner) => {
                init.push(InitNested::List(field.is_nullable));
                let (mut nested, array, ptm) = columns_to_iter_recursive(
                    columns,
                    types,
                    inner.as_ref().clone(),
                    init,
                    filter,
                )?;
                let array = create_list(field.dtype().clone(), &mut nested, array);
                Ok((nested, array, ptm))
            },
            ArrowDataType::FixedSizeList(inner, width) => {
                init.push(InitNested::FixedSizeList(field.is_nullable, *width));
                let (mut nested, array, ptm) = columns_to_iter_recursive(
                    columns,
                    types,
                    inner.as_ref().clone(),
                    init,
                    filter,
                )?;
                let array = create_list(field.dtype().clone(), &mut nested, array);
                Ok((nested, array, ptm))
            },
            ArrowDataType::Struct(fields) => {
                // This definitely does not support Filter predicate yet.
                assert!(!matches!(&filter, Some(Filter::Predicate(_))));

                // @NOTE:
                // We go back to front here, because we constantly split off the end of the array
                // to grab the relevant columns and types.
                //
                // Is this inefficient? Yes. Is this how we are going to do it for now? Yes.

                let Some(last_field) = fields.last() else {
                    return Err(ParquetError::not_supported("Struct has zero fields"));
                };

                let field_to_nested_array =
                    |mut init: Vec<InitNested>,
                     columns: &mut Vec<BasicDecompressor>,
                     types: &mut Vec<&PrimitiveType>,
                     struct_field: &Field| {
                        init.push(InitNested::Struct(field.is_nullable));
                        let n = n_columns(&struct_field.dtype);
                        let columns = columns.split_off(columns.len() - n);
                        let types = types.split_off(types.len() - n);

                        columns_to_iter_recursive(
                            columns,
                            types,
                            struct_field.clone(),
                            init,
                            filter.clone(),
                        )
                    };

                let (mut nested, last_array, _) =
                    field_to_nested_array(init.clone(), &mut columns, &mut types, last_field)?;
                debug_assert!(matches!(nested.last().unwrap(), NestedContent::Struct));
                let (length, _, struct_validity) = nested.pop().unwrap();

                let mut field_arrays = Vec::<Box<dyn Array>>::with_capacity(fields.len());
                field_arrays.push(last_array);

                for field in fields.iter().rev().skip(1) {
                    let (mut _nested, array, _) =
                        field_to_nested_array(init.clone(), &mut columns, &mut types, field)?;

                    #[cfg(debug_assertions)]
                    {
                        debug_assert!(matches!(_nested.last().unwrap(), NestedContent::Struct));
                        debug_assert_eq!(
                            _nested.pop().unwrap().2.and_then(freeze_validity),
                            struct_validity.clone().and_then(freeze_validity),
                        );
                    }

                    field_arrays.push(array);
                }

                field_arrays.reverse();
                let struct_validity = struct_validity.and_then(freeze_validity);

                Ok((
                    nested,
                    StructArray::new(
                        ArrowDataType::Struct(fields.clone()),
                        length,
                        field_arrays,
                        struct_validity,
                    )
                    .to_boxed(),
                    Bitmap::new(),
                ))
            },
            ArrowDataType::Map(inner, _) => {
                init.push(InitNested::List(field.is_nullable));
                let (mut nested, array, ptm) = columns_to_iter_recursive(
                    columns,
                    types,
                    inner.as_ref().clone(),
                    init,
                    filter,
                )?;
                let array = create_map(field.dtype().clone(), &mut nested, array);
                Ok((nested, array, ptm))
            },

            ArrowDataType::Dictionary(key_type, value_type, _) => {
                // @note: this should only hit in two cases:
                // - polars enum's and categorical's
                // - int -> string which can be turned into categoricals
                assert!(matches!(value_type.as_ref(), ArrowDataType::Utf8View));

                init.push(InitNested::Primitive(field.is_nullable));

                if field.metadata.as_ref().is_none_or(|md| {
                    !md.contains_key(DTYPE_ENUM_VALUES) && !md.contains_key(DTYPE_CATEGORICAL)
                }) {
                    let (nested, arr, ptm) = PageDecoder::new(
                        columns.pop().unwrap(),
                        ArrowDataType::Utf8View,
                        binview::BinViewDecoder::new_string(),
                        Some(init),
                    )?
                    .collect_nested(filter)?;

                    let arr = polars_compute::cast::cast(
                        arr.as_ref(),
                        field.dtype(),
                        CastOptionsImpl::default(),
                    )
                    .unwrap();

                    Ok((nested, arr, ptm))
                } else {
                    assert!(matches!(key_type, IntegerType::UInt32));

                    let (nested, arr, ptm) = PageDecoder::new(
                        columns.pop().unwrap(),
                        field.dtype().clone(),
                        CategoricalDecoder::new(),
                        Some(init),
                    )?
                    .collect_boxed(filter)?;

                    Ok((nested.unwrap(), arr, ptm))
                }
            },
            other => Err(ParquetError::not_supported(format!(
                "Deserializing type {other:?} from parquet"
            ))),
        }
    }
}
