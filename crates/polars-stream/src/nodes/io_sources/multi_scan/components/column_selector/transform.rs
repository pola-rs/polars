use arrow::array::LIST_VALUES_NAME;
use polars_array::PlBitmap;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::chunked_array::flags::StatisticsFlags;
use polars_core::prelude::{Column, DataType, IntoColumn, PlArrayRef};
use polars_core::series::{IntoSeries, Series};
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::multi_scan::components::column_selector::ColumnSelector;

#[derive(Debug, Clone)]
pub enum ColumnTransform {
    /// Cast the column to a dtype.
    Cast {
        dtype: DataType,
        options: CastOptions,
    },
    /// Set the name of the column.
    Rename { name: PlSmallStr },
    /// Construct a struct column by applying column selectors onto the field arrays.
    StructFieldsMapping {
        field_selectors: Box<[ColumnSelector]>,
    },
    /// Construct a list column by applying column selectors onto the values array.
    ListValuesMapping { values_selector: ColumnSelector },
    #[cfg(feature = "dtype-array")]
    FixedSizeListValuesMapping { values_selector: ColumnSelector },
    /// Construct a map column by applying a column selector onto the value child.
    #[cfg(feature = "dtype-map")]
    MapValuesMapping { value_selector: ColumnSelector },
}

impl ColumnTransform {
    pub fn into_selector(self, input_selector: ColumnSelector) -> ColumnSelector {
        ColumnSelector::Transformed(Box::new((input_selector, self)))
    }

    pub fn apply_transform(&self, input: Column) -> PolarsResult<Column> {
        use ColumnTransform as TF;

        let out = match self {
            TF::Cast { dtype, options } => {
                // note: strict casts are only sound for non-nested dtypes here;
                // recursion currently does not propagate NULLs across nesting levels
                debug_assert!(!matches!(options, CastOptions::Strict) || !dtype.is_nested());

                input.cast_with_options(dtype, *options)?
            },

            TF::Rename { name } => input.with_name(name.clone()),

            TF::StructFieldsMapping { field_selectors } => {
                use polars_core::prelude::StructChunked;

                let input_s = input._get_backing_series();
                let struct_ca = input_s.struct_().unwrap();
                let field_columns: Vec<Column> = struct_ca.fields_as_columns();

                let field_columns: Vec<Column> = field_selectors
                    .iter()
                    .map(|x| x.select_from_columns(&field_columns, struct_ca.len()))
                    .collect::<PolarsResult<_>>()?;

                input._to_new_from_backing(
                    StructChunked::from_columns(
                        struct_ca.name().clone(),
                        struct_ca.len(),
                        &field_columns,
                    )?
                    .with_outer_validity(struct_ca.rechunk_validity())
                    .into_series(),
                )
            },

            #[cfg(feature = "dtype-map")]
            TF::MapValuesMapping { value_selector } => {
                let input_s = input._get_backing_series();
                let map_ca = input_s.map().unwrap();

                let values: Column = map_ca.values().into_column();
                let len = values.len();
                let values = value_selector.select_from_columns(&[values], len)?;

                input._to_new_from_backing(
                    map_ca
                        .with_values(values.as_materialized_series())?
                        .into_series(),
                )
            },

            TF::ListValuesMapping { values_selector } => {
                use polars_core::prelude::{ListChunked, PlListArray};

                let input_list_ca = input._get_backing_series().list().unwrap().clone();

                let values_dtype = {
                    let DataType::List(inner) = input_list_ca.dtype() else {
                        unreachable!()
                    };
                    inner.as_ref()
                };

                let mut values_output_dtype = None;

                let mut out_chunks: Vec<PlArrayRef> =
                    Vec::with_capacity(input_list_ca.chunks().len());

                for list_arr in input_list_ca.downcast_iter() {
                    let values: Column = unsafe {
                        Series::from_chunks_and_dtype_unchecked(
                            LIST_VALUES_NAME,
                            vec![list_arr.values().to_boxed()],
                            values_dtype,
                        )
                    }
                    .into_column();
                    let len = values.len();

                    let values: Column = values_selector.select_from_columns(&[values], len)?;

                    if values_output_dtype.is_none() {
                        values_output_dtype = Some(values.dtype().clone());
                    }

                    let values: PlArrayRef = values
                        .as_materialized_series()
                        .rechunk()
                        .into_chunks()
                        .pop()
                        .unwrap();

                    // The offsets and the mask are handed over in whatever representation each is
                    // in: only the values were mapped, and that leaves their number untouched, so
                    // offsets holding the one range every element covers still hold it.
                    let validity = list_arr.validity().map(PlBitmap::from);
                    let offsets_are_scalar = list_arr.offsets_are_scalar();
                    let (_, offsets, length, _) = list_arr.clone().into_inner();

                    let list_arr = match offsets_are_scalar {
                        true => PlListArray::new_broadcast(values, offsets, length, validity),
                        false => PlListArray::new(values, offsets, length, validity),
                    };

                    out_chunks.push(Box::new(list_arr))
                }

                let mut out =
                    unsafe { ListChunked::from_chunks(input_list_ca.name().clone(), out_chunks) };

                // Ensure logical types are restored.
                // SAFETY: chunks retain the selector's output dtype and valid values.
                unsafe { out.set_inner_dtype(values_output_dtype.unwrap()) };

                // Casts on the values should not affect outer NULLs.
                out.retain_flags_from(&input_list_ca, StatisticsFlags::CAN_FAST_EXPLODE_LIST);

                input._to_new_from_backing(out.into_series())
            },

            #[cfg(feature = "dtype-array")]
            TF::FixedSizeListValuesMapping { values_selector } => {
                use polars_core::prelude::{ArrayChunked, PlFixedSizeListArray};

                let input_array_ca = input._get_backing_series().array().unwrap().clone();

                let values_dtype = {
                    let DataType::Array(inner, _) = input_array_ca.dtype() else {
                        unreachable!()
                    };
                    inner.as_ref()
                };

                let mut values_output_dtype = None;

                let mut out_chunks: Vec<PlArrayRef> =
                    Vec::with_capacity(input_array_ca.chunks().len());

                for fixed_size_list_arr in input_array_ca.downcast_iter() {
                    let values: Column = unsafe {
                        Series::from_chunks_and_dtype_unchecked(
                            LIST_VALUES_NAME,
                            vec![fixed_size_list_arr.values().to_boxed()],
                            values_dtype,
                        )
                    }
                    .into_column();
                    let len = values.len();

                    let values: Column = values_selector.select_from_columns(&[values], len)?;

                    if values_output_dtype.is_none() {
                        values_output_dtype = Some(values.dtype().clone());
                    }

                    let values: PlArrayRef = values
                        .as_materialized_series()
                        .rechunk()
                        .into_chunks()
                        .pop()
                        .unwrap();

                    // The width and the mask are handed over in whatever representation each is
                    // in: only the values were mapped, and that leaves their number untouched, so
                    // values holding the one list every element reads still hold it.
                    let validity = fixed_size_list_arr.validity().map(PlBitmap::from);
                    let values_are_scalar = fixed_size_list_arr.values_are_scalar();
                    let (_, width, length, _) = fixed_size_list_arr.clone().into_inner();

                    let fixed_size_list_arr = match values_are_scalar {
                        true => {
                            PlFixedSizeListArray::new_broadcast(values, width, length, validity)
                        },
                        false => PlFixedSizeListArray::new(values, width, length, validity),
                    };

                    out_chunks.push(Box::new(fixed_size_list_arr))
                }

                // The chunks carry no inner type, so the array is built with its dtype directly,
                // which restores the logical type of the values as well.
                let out = unsafe {
                    ArrayChunked::from_chunks_and_dtype(
                        input_array_ca.name().clone(),
                        out_chunks,
                        DataType::Array(
                            Box::new(values_output_dtype.unwrap()),
                            input_array_ca.width(),
                        ),
                    )
                };

                input._to_new_from_backing(out.into_series())
            },
        };

        Ok(out)
    }
}
