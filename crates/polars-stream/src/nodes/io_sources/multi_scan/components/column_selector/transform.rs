use arrow::array::{Array, LIST_VALUES_NAME};
use arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_core::chunked_array::cast::CastOptions;
use polars_core::chunked_array::flags::StatisticsFlags;
use polars_core::prelude::{Column, DataType, IntoColumn};
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
}

impl ColumnTransform {
    pub fn into_selector(self, input_selector: ColumnSelector) -> ColumnSelector {
        ColumnSelector::Transformed(Box::new((input_selector, self)))
    }

    pub fn apply_transform(&self, input: Column) -> PolarsResult<Column> {
        use ColumnTransform as TF;

        let out = match self {
            TF::Cast { dtype, options } => {
                // Recursion currently does not propagate NULLs across nesting levels.
                debug_assert!(!matches!(options, CastOptions::Strict));

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

            TF::ListValuesMapping { values_selector } => {
                use polars_core::prelude::{LargeListArray, ListChunked};

                let input_list_ca = input._get_backing_series().list().unwrap().clone();

                let values_dtype = {
                    let DataType::List(inner) = input_list_ca.dtype() else {
                        unreachable!()
                    };
                    inner.as_ref()
                };

                let mut values_output_dtype = None;

                let mut out_chunks: Vec<Box<dyn Array>> =
                    Vec::with_capacity(input_list_ca.chunks().len());

                for list_arr in input_list_ca.downcast_iter() {
                    let values: Box<dyn Array> = list_arr.values().clone();
                    let values: Column = unsafe {
                        Series::from_chunks_and_dtype_unchecked(
                            LIST_VALUES_NAME,
                            vec![values],
                            values_dtype,
                        )
                    }
                    .into_column();
                    let len = values.len();

                    let values: Column = values_selector.select_from_columns(&[values], len)?;

                    if values_output_dtype.is_none() {
                        values_output_dtype = Some(values.dtype().clone());
                    }

                    let values: Box<dyn Array> = values
                        .as_materialized_series()
                        .rechunk()
                        .into_chunks()
                        .pop()
                        .unwrap();

                    let list_arr = LargeListArray::new(
                        ArrowDataType::LargeList(Box::new(ArrowField::new(
                            LIST_VALUES_NAME,
                            values.dtype().clone(),
                            true,
                        ))),
                        list_arr.offsets().clone(),
                        values,
                        list_arr.validity().cloned(),
                    );

                    out_chunks.push(list_arr.boxed())
                }

                let mut out =
                    unsafe { ListChunked::from_chunks(input_list_ca.name().clone(), out_chunks) };

                // Ensure logical types are restored.
                out.set_inner_dtype(values_output_dtype.unwrap());

                // Casts on the values should not affect outer NULLs.
                out.retain_flags_from(&input_list_ca, StatisticsFlags::CAN_FAST_EXPLODE_LIST);

                input._to_new_from_backing(out.into_series())
            },

            #[cfg(feature = "dtype-array")]
            TF::FixedSizeListValuesMapping { values_selector } => {
                use arrow::array::FixedSizeListArray;
                use polars_core::prelude::ArrayChunked;

                let input_array_ca = input._get_backing_series().array().unwrap().clone();

                let values_dtype = {
                    let DataType::Array(inner, _) = input_array_ca.dtype() else {
                        unreachable!()
                    };
                    inner.as_ref()
                };

                let mut values_output_dtype = None;

                let mut out_chunks: Vec<Box<dyn Array>> =
                    Vec::with_capacity(input_array_ca.chunks().len());

                for fixed_size_list_arr in input_array_ca.downcast_iter() {
                    let values: Box<dyn Array> = fixed_size_list_arr.values().clone();
                    let values: Column = unsafe {
                        Series::from_chunks_and_dtype_unchecked(
                            LIST_VALUES_NAME,
                            vec![values],
                            values_dtype,
                        )
                    }
                    .into_column();
                    let len = values.len();

                    let values: Column = values_selector.select_from_columns(&[values], len)?;

                    if values_output_dtype.is_none() {
                        values_output_dtype = Some(values.dtype().clone());
                    }

                    let values: Box<dyn Array> = values
                        .as_materialized_series()
                        .rechunk()
                        .into_chunks()
                        .pop()
                        .unwrap();

                    let fixed_size_list_arr = FixedSizeListArray::new(
                        ArrowDataType::FixedSizeList(
                            Box::new(ArrowField::new(
                                LIST_VALUES_NAME,
                                values.dtype().clone(),
                                true,
                            )),
                            fixed_size_list_arr.size(),
                        ),
                        fixed_size_list_arr.len(),
                        values,
                        fixed_size_list_arr.validity().cloned(),
                    );

                    out_chunks.push(fixed_size_list_arr.boxed())
                }

                let mut out =
                    unsafe { ArrayChunked::from_chunks(input_array_ca.name().clone(), out_chunks) };

                // Ensure logical types are restored.
                out.set_inner_dtype(values_output_dtype.unwrap());

                input._to_new_from_backing(out.into_series())
            },
        };

        Ok(out)
    }
}
