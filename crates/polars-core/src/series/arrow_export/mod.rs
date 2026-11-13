macro_rules! bail_unhandled_arrow_conversion_dtype_pair {
    ($input_pl_dtype:expr, $output_arrow_field:expr) => {{
        return Err(
            $crate::series::arrow_export::unhandled_arrow_conversion_dtype_pair_err(
                $input_pl_dtype,
                $output_arrow_field,
            ),
        );
    }};
}

#[cfg(feature = "dtype-categorical")]
pub mod categorical;

use std::borrow::Cow;
use std::sync::Arc;

use polars_arrow::array::{FixedSizeListArray, StructArray};
use polars_compute::cast::cast_unchecked;
use polars_compute::rebuild_list::rebuild_list_shallow;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_ensure, polars_err};

use crate::prelude::{
    Array, ArrayRef, ArrowDataType, ArrowField, BinaryViewArray, CompatLevel, DataType, ListArray,
    PlSmallStr, PrimitiveArray, Series,
};

fn unhandled_arrow_conversion_dtype_pair_err(
    input_pl_dtype: &DataType,
    output_arrow_field: &ArrowField,
) -> PolarsError {
    polars_err!(
        InvalidOperation:
        "to_arrow() conversion failed: cannot convert \
        ({input_pl_dtype:?}) to ({output_arrow_field:?})",
    )
}

/// Downcasts to a primitive array, boxes it, then sets its dtype.
macro_rules! primitive_to_boxed_with_logical {
    ($array:expr, $physical:ty, $logical_arrow_dtype:expr) => {{
        let arr: &PrimitiveArray<$physical> = $array.as_any().downcast_ref().unwrap();
        arr.clone().to($logical_arrow_dtype).to_boxed()
    }};
}

/// Drop the entries no live row owns and rebase the offsets onto the rest; `None` if the
/// chunk already spans exactly its child.
fn normalize_map_entries(arr: &ListArray<i64>) -> Option<ListArray<i64>> {
    #[cfg(feature = "dtype-map")]
    if let Some(compacted) = crate::chunked_array::logical::compact_null_rows_chunk(arr) {
        return Some(compacted);
    }

    trim_list_child(arr)
}

/// Slice the child of `arr` to the elements its offsets span; `None` if it already is.
fn trim_list_child(arr: &ListArray<i64>) -> Option<ListArray<i64>> {
    let offsets = arr.offsets();
    let first = *offsets.first() as usize;
    let len = offsets.range() as usize;
    if first == 0 && len == arr.values().len() {
        return None;
    }
    Some(rebuild_list_shallow(
        arr,
        arr.dtype().clone(),
        arr.values().sliced(first, len),
    ))
}

/// Whether exporting to `dtype` builds an arrow MAP array, whose offsets are `i32`.
pub fn contains_arrow_map(dtype: &ArrowDataType) -> bool {
    match dtype {
        ArrowDataType::Map(_, _) => true,
        ArrowDataType::LargeList(field) | ArrowDataType::FixedSizeList(field, _) => {
            contains_arrow_map(field.dtype())
        },
        ArrowDataType::Struct(fields) => fields.iter().any(|f| contains_arrow_map(f.dtype())),
        ArrowDataType::Extension(ext) => contains_arrow_map(&ext.inner),
        _ => false,
    }
}

/// Whether some MAP array exported from `array` to `dtype` could exceed `i32::MAX` entries.
///
/// Checks each MAP's whole child, an upper bound for any slice of `array`.
fn may_overflow_arrow_map(array: &dyn Array, dtype: &ArrowDataType) -> bool {
    match dtype {
        ArrowDataType::Map(entries, _) | ArrowDataType::LargeList(entries) => {
            let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            (matches!(dtype, ArrowDataType::Map(_, _)) && arr.offsets().range() > i32::MAX as i64)
                || may_overflow_arrow_map(arr.values().as_ref(), entries.dtype())
        },
        ArrowDataType::FixedSizeList(item, _) => {
            let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
            may_overflow_arrow_map(arr.values().as_ref(), item.dtype())
        },
        ArrowDataType::Struct(fields) => {
            let arr: &StructArray = array.as_any().downcast_ref().unwrap();
            arr.values()
                .iter()
                .zip(fields)
                .any(|(values, f)| may_overflow_arrow_map(values.as_ref(), f.dtype()))
        },
        ArrowDataType::Extension(ext) => may_overflow_arrow_map(array, &ext.inner),
        _ => false,
    }
}

/// Push, for every MAP exported from `array` to `dtype` that may overflow, the running count
/// of its entries over the rows of `array` (`array.len() + 1` values, starting at 0).
fn push_map_entry_counts(array: &dyn Array, dtype: &ArrowDataType, out: &mut Vec<Vec<i64>>) {
    if !may_overflow_arrow_map(array, dtype) {
        return;
    }

    match dtype {
        ArrowDataType::Map(entries, _) | ArrowDataType::LargeList(entries) => {
            let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            let offsets = arr.offsets().as_slice();
            let first = offsets[0];
            if matches!(dtype, ArrowDataType::Map(_, _)) && arr.offsets().range() > i32::MAX as i64
            {
                out.push(offsets.iter().map(|o| o - first).collect());
            }

            let nested_start = out.len();
            push_map_entry_counts(arr.values().as_ref(), entries.dtype(), out);
            for counts in &mut out[nested_start..] {
                let base = counts[first as usize];
                *counts = offsets.iter().map(|&o| counts[o as usize] - base).collect();
            }
        },
        ArrowDataType::FixedSizeList(item, width) => {
            let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
            let nested_start = out.len();
            push_map_entry_counts(arr.values().as_ref(), item.dtype(), out);
            for counts in &mut out[nested_start..] {
                *counts = (0..=arr.len()).map(|i| counts[i * width]).collect();
            }
        },
        ArrowDataType::Struct(fields) => {
            let arr: &StructArray = array.as_any().downcast_ref().unwrap();
            for (values, f) in arr.values().iter().zip(fields) {
                push_map_entry_counts(values.as_ref(), f.dtype(), out);
            }
        },
        ArrowDataType::Extension(ext) => push_map_entry_counts(array, &ext.inner, out),
        _ => {},
    }
}

/// Row bounds (from 0 to `len`) that cut the `len` rows of `columns` into pieces whose arrow
/// MAP arrays each fit in `i32` offsets.
///
/// Errors if a single row has more MAP entries than fit.
pub fn arrow_export_row_bounds<'a>(
    columns: impl IntoIterator<Item = (&'a dyn Array, &'a ArrowDataType)>,
    len: usize,
) -> PolarsResult<Vec<usize>> {
    let mut counts = Vec::new();
    for (array, dtype) in columns {
        if contains_arrow_map(dtype) {
            push_map_entry_counts(array, dtype, &mut counts);
        }
    }

    if counts.is_empty() {
        return Ok(vec![0, len]);
    }

    let max = i32::MAX as i64;
    let mut bounds = vec![0];
    let mut start = 0;
    for row in 0..len {
        if counts.iter().all(|c| c[row + 1] - c[start] <= max) {
            continue;
        }
        if let Some(c) = counts.iter().find(|c| c[row + 1] - c[row] > max) {
            polars_bail!(
                InvalidOperation:
                "to_arrow() conversion failed: a single row has {} map entries, more than \
                the i32 offsets of the arrow MAP type can hold",
                c[row + 1] - c[row],
            );
        }
        bounds.push(row);
        start = row;
    }
    bounds.push(len);

    Ok(bounds)
}

fn ensure_no_nulls(array: &dyn Array) -> PolarsResult<()> {
    polars_ensure!(
        !array.has_nulls(),
        SchemaMismatch:
        "to_arrow() conversion failed: nullable is false but array contained {} NULLs (arrow dtype: {:?})",
        array.null_count(), array.dtype(),
    );

    Ok(())
}

impl Series {
    /// Export this Series to an arrow array. The dtype of the returned array will be chosen
    /// according to the provided `compat_level`.
    ///
    /// # Panics
    /// Panics if the chunk cannot be exported, see [`Series::try_to_arrow`].
    pub fn to_arrow(&self, chunk_idx: usize, compat_level: CompatLevel) -> ArrayRef {
        self.try_to_arrow(chunk_idx, compat_level).unwrap()
    }

    /// Export this Series to an arrow array, or error if the chunk does not fit the arrow
    /// type, e.g. a Map chunk with more than `i32::MAX` entries.
    pub fn try_to_arrow(
        &self,
        chunk_idx: usize,
        compat_level: CompatLevel,
    ) -> PolarsResult<ArrayRef> {
        self.to_arrow_with_field(
            chunk_idx,
            Cow::Owned(
                self.dtype()
                    .to_arrow_field(self.name().clone(), compat_level),
            ),
            true,
        )
    }

    /// Export a chunk of this Series as one or more arrow arrays, split at rows so that each
    /// arrow MAP array fits in `i32` offsets.
    pub fn try_to_arrow_split(
        &self,
        chunk_idx: usize,
        compat_level: CompatLevel,
    ) -> PolarsResult<Vec<ArrayRef>> {
        let chunk = self.chunks()[chunk_idx].as_ref();
        let arrow_dtype = self.dtype().to_arrow(compat_level);
        let bounds = arrow_export_row_bounds([(chunk, &arrow_dtype)], chunk.len())?;
        if bounds.len() == 2 {
            return Ok(vec![self.try_to_arrow(chunk_idx, compat_level)?]);
        }

        let chunk = self.select_chunk(chunk_idx);
        bounds
            .windows(2)
            .map(|w| {
                chunk
                    .slice(w[0] as i64, w[1] - w[0])
                    .try_to_arrow(0, compat_level)
            })
            .collect()
    }

    /// Export this Series to an arrow array. The dtype of the returned array will match the
    /// provided arrow field. Returns an error if this Series cannot be exported to the arrow field.
    pub fn to_arrow_with_field<'a>(
        &self,
        chunk_idx: usize,
        output_arrow_field: Cow<'a, ArrowField>,
        skip_attach_pl_metadata: bool,
    ) -> PolarsResult<ArrayRef> {
        ToArrowConverter {
            skip_attach_pl_metadata,
            #[cfg(feature = "dtype-categorical")]
            categorical_converter: {
                let mut categorical_converter =
                    crate::series::arrow_export::categorical::CategoricalToArrowConverter {
                        converters: Default::default(),
                        persist_remap: false,
                    };

                categorical_converter.initialize(self.dtype());

                categorical_converter
            },
        }
        .array_to_arrow(
            self.chunks().get(chunk_idx).unwrap().as_ref(),
            self.dtype(),
            output_arrow_field,
        )
    }
}

/// Low-level converter that exports `ArrayRef`s from Polars Series to arrow arrays.
///
/// This can be held to perform repeated categorical exports with persisted indices to ensure
/// the exported chunks use the same set of indices.
pub struct ToArrowConverter {
    /// If the `arrow_field` being passed was generated by `DataType::to_arrow_field`,
    /// it will already have polars metadata.
    pub skip_attach_pl_metadata: bool,
    #[cfg(feature = "dtype-categorical")]
    pub categorical_converter:
        crate::series::arrow_export::categorical::CategoricalToArrowConverter,
}

impl ToArrowConverter {
    /// Returns an error if `output_arrow_field` was provided and does not match the output data type.
    pub fn array_to_arrow<'a>(
        &mut self,
        array: &dyn Array,
        dtype: &DataType,
        arrow_field: Cow<'a, ArrowField>,
    ) -> PolarsResult<Box<dyn Array>> {
        let nullable = arrow_field.is_nullable;
        let out = self.array_to_arrow_impl(array, dtype, arrow_field)?;

        if !nullable {
            ensure_no_nulls(array)?
        }

        Ok(out)
    }

    fn array_to_arrow_impl<'a>(
        &mut self,
        array: &dyn Array,
        polars_dtype: &DataType,
        arrow_field: Cow<'a, ArrowField>,
    ) -> PolarsResult<Box<dyn Array>> {
        // We perform additional steps where necessary. E.g.
        // * If we are exporting a logical type, set the array dtype to the corresponding arrow logical type.
        // * Attach field metadata where necessary (e.g. for categorical and extension types).
        Ok(match (polars_dtype, arrow_field.dtype()) {
            #[cfg(feature = "dtype-struct")]
            (DataType::Struct(struct_fields), ArrowDataType::Struct(arrow_struct_fields)) => {
                let arr: &StructArray = array.as_any().downcast_ref().unwrap();

                polars_ensure!(
                    arrow_struct_fields.len() == arr.fields().len()
                    && arrow_struct_fields
                        .iter()
                        .zip(arr.fields())
                        .all(|(l, r)| l.name() == r.name()),
                    SchemaMismatch:
                    "to_arrow() conversion failed: struct field names mismatch: {:?} != expected: {:?}",
                    arrow_field.dtype(), arr.dtype()
                );

                let mut arrow_dtype = to_owned_dtype(arrow_field);

                let ArrowDataType::Struct(arrow_struct_fields) = &mut arrow_dtype else {
                    unreachable!()
                };

                self.attach_pl_field_metadata(
                    struct_fields
                        .iter()
                        .map(|x| x.dtype())
                        .zip(arrow_struct_fields.iter_mut()),
                );

                let values: Vec<ArrayRef> = arr
                    .values()
                    .iter()
                    .zip(struct_fields.iter())
                    .zip(arrow_struct_fields.iter())
                    .map(|((values, pl_field), arrow_field)| {
                        self.array_to_arrow(
                            values.as_ref(),
                            pl_field.dtype(),
                            Cow::Borrowed(arrow_field),
                        )
                    })
                    .collect::<PolarsResult<_>>()?;

                let arr =
                    StructArray::try_new(arrow_dtype, arr.len(), values, arr.validity().cloned())?;

                Box::new(arr)
            },
            (DataType::List(item_dtype), ArrowDataType::LargeList(arrow_item_field)) => {
                let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
                // Export only the child this list spans, so that a slice of a list of maps has
                // only its own map entries.
                let trimmed = contains_arrow_map(arrow_item_field.dtype())
                    .then(|| trim_list_child(arr))
                    .flatten();
                let arr = trimmed.as_ref().unwrap_or(arr);

                let mut arrow_dtype = to_owned_dtype(arrow_field);

                let ArrowDataType::LargeList(arrow_item_field) = &mut arrow_dtype else {
                    unreachable!()
                };

                self.attach_pl_field_metadata(std::iter::once((
                    item_dtype.as_ref(),
                    arrow_item_field.as_mut(),
                )));

                let new_values = self.array_to_arrow(
                    arr.values().as_ref(),
                    item_dtype,
                    Cow::Borrowed(arrow_item_field.as_ref()),
                )?;

                let arr = ListArray::<i64>::new(
                    arrow_dtype,
                    arr.offsets().clone(),
                    new_values,
                    arr.validity().cloned(),
                );

                Box::new(arr)
            },
            #[cfg(feature = "dtype-map")]
            (DataType::Map(_, _), ArrowDataType::Map(_, _)) => {
                let entries_dtype = polars_dtype.map_entries_dtype().unwrap();
                self.map_array_to_arrow(array, &entries_dtype, arrow_field)?
            },
            (DataType::List(entries_dtype), ArrowDataType::Map(_, _)) => {
                self.map_array_to_arrow(array, entries_dtype, arrow_field)?
            },
            #[cfg(feature = "dtype-map")]
            (DataType::Map(_, _), ArrowDataType::LargeList(_)) => {
                let storage_dtype = polars_dtype.map_storage_dtype().unwrap();
                self.array_to_arrow_impl(array, &storage_dtype, arrow_field)?
            },
            #[cfg(feature = "dtype-array")]
            (DataType::Array(item_dtype, width), ArrowDataType::FixedSizeList(_, arrow_width)) => {
                let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();

                polars_ensure!(
                    *arrow_width == *width,
                    SchemaMismatch:
                    "to_arrow() conversion failed: fixed-size list width mismatch \
                    ({arrow_width:?} != expected: {width:?})"
                );

                let mut arrow_dtype = to_owned_dtype(arrow_field);

                let ArrowDataType::FixedSizeList(arrow_item_field, _) = &mut arrow_dtype else {
                    unreachable!()
                };

                self.attach_pl_field_metadata(std::iter::once((
                    item_dtype.as_ref(),
                    arrow_item_field.as_mut(),
                )));

                let new_values = self.array_to_arrow(
                    arr.values().as_ref(),
                    item_dtype,
                    Cow::Borrowed(arrow_item_field.as_ref()),
                )?;

                let arr = FixedSizeListArray::new(
                    arrow_dtype,
                    arr.len(),
                    new_values,
                    arr.validity().cloned(),
                );

                Box::new(arr)
            },
            #[cfg(feature = "dtype-categorical")]
            (DataType::Categorical(_, _) | DataType::Enum(_, _), _) => {
                self.categorical_converter.array_to_arrow(
                    array,
                    polars_dtype,
                    arrow_field.as_ref(),
                )?
            },
            #[cfg(feature = "dtype-date")]
            (DataType::Date, ArrowDataType::Date32) => {
                primitive_to_boxed_with_logical!(array, i32, ArrowDataType::Date32)
            },
            #[cfg(feature = "dtype-datetime")]
            (DataType::Datetime(tu, tz), ArrowDataType::Timestamp(atu, atz)) => {
                use crate::prelude::TimeZone;

                let matching = atu == &tu.to_arrow()
                    && TimeZone::eq_none_as_utc(
                        TimeZone::opt_try_new(atz.clone())?.as_ref(),
                        tz.as_ref(),
                    );

                if !matching {
                    bail_unhandled_arrow_conversion_dtype_pair!(polars_dtype, &arrow_field)
                }

                primitive_to_boxed_with_logical!(array, i64, to_owned_dtype(arrow_field))
            },
            #[cfg(feature = "dtype-duration")]
            (DataType::Duration(tu), ArrowDataType::Duration(atu)) => {
                let matching = atu == &tu.to_arrow();

                if !matching {
                    bail_unhandled_arrow_conversion_dtype_pair!(polars_dtype, &arrow_field)
                }

                primitive_to_boxed_with_logical!(array, i64, to_owned_dtype(arrow_field))
            },
            #[cfg(feature = "dtype-time")]
            (DataType::Time, ArrowDataType::Time64(crate::prelude::ArrowTimeUnit::Nanosecond)) => {
                primitive_to_boxed_with_logical!(array, i64, to_owned_dtype(arrow_field))
            },
            #[cfg(feature = "dtype-time")]
            (DataType::Time, ArrowDataType::Time64(crate::prelude::ArrowTimeUnit::Microsecond)) => {
                use polars_compute::cast::time64ns_to_time64us;

                let array: &PrimitiveArray<i64> = array.as_any().downcast_ref().unwrap();

                time64ns_to_time64us(array).boxed()
            },
            #[cfg(feature = "dtype-decimal")]
            (DataType::Decimal(prec, scale), ArrowDataType::Decimal(a_prec, a_scale)) => {
                let matching = *a_prec == *prec && *a_scale == *scale;

                if !matching {
                    bail_unhandled_arrow_conversion_dtype_pair!(polars_dtype, &arrow_field)
                }

                primitive_to_boxed_with_logical!(array, i128, to_owned_dtype(arrow_field))
            },
            #[cfg(feature = "object")]
            (DataType::Object(_), ArrowDataType::FixedSizeBinary(8)) => {
                use crate::chunked_array::object::builder::object_series_to_arrow_array;

                let out = object_series_to_arrow_array(&unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        PlSmallStr::EMPTY,
                        vec![array.to_boxed()],
                        polars_dtype,
                    )
                });

                assert_eq!(out.dtype(), &ArrowDataType::FixedSizeBinary(8));

                out
            },
            (DataType::String, ArrowDataType::Utf8View) => array.to_boxed(),
            (DataType::String, ArrowDataType::LargeUtf8) => {
                cast_unchecked(array, &ArrowDataType::LargeUtf8).unwrap()
            },
            (DataType::Binary, ArrowDataType::BinaryView) => array.to_boxed(),
            (DataType::Binary, ArrowDataType::LargeBinary) => {
                cast_unchecked(array, &ArrowDataType::LargeBinary).unwrap()
            },
            (DataType::Binary, ArrowDataType::FixedSizeBinary(row_width)) => {
                use polars_compute::cast::binview_to_fixed_binary;

                let array: &BinaryViewArray = array.as_any().downcast_ref().unwrap();

                binview_to_fixed_binary(array, *row_width)?.boxed()
            },
            (DataType::Binary, ArrowDataType::Extension(_)) => {
                let arrow_dtype = to_owned_dtype(arrow_field);

                let ArrowDataType::Extension(ext_type) = &arrow_dtype else {
                    unreachable!()
                };

                let storage_field =
                    ArrowField::new(ext_type.name.clone(), ext_type.inner.clone(), true);

                let mut array =
                    self.array_to_arrow(array, &DataType::Binary, Cow::Owned(storage_field))?;

                *array.dtype_mut() = arrow_dtype;

                array.to_boxed()
            },
            #[cfg(feature = "dtype-extension")]
            (
                DataType::Extension(pl_ext_type, storage_dtype),
                ArrowDataType::Extension(arrow_ext_type),
            ) => {
                use polars_arrow::datatypes::ExtensionType;

                let ExtensionType {
                    name,
                    inner: _,
                    metadata,
                } = arrow_ext_type.as_ref();

                if name != pl_ext_type.name().as_ref() {
                    bail_unhandled_arrow_conversion_dtype_pair!(polars_dtype, &arrow_field)
                }

                match (
                    metadata.as_deref(),
                    pl_ext_type.serialize_metadata().as_deref(),
                ) {
                    (Some("") | None, Some("") | None) => {},
                    (l, r) => {
                        if l != r {
                            bail_unhandled_arrow_conversion_dtype_pair!(polars_dtype, &arrow_field)
                        }
                    },
                };

                let arrow_dtype = to_owned_dtype(arrow_field);

                let ArrowDataType::Extension(arrow_ext_type) = &arrow_dtype else {
                    unreachable!()
                };

                let storage_arrow_field = ArrowField::new(
                    arrow_ext_type.name.clone(),
                    arrow_ext_type.inner.clone(),
                    true,
                );

                let mut arr =
                    self.array_to_arrow(array, storage_dtype, Cow::Owned(storage_arrow_field))?;

                *arr.dtype_mut() = arrow_dtype;

                arr
            },
            (pl_dtype, arrow_dtype) => {
                if array.dtype() != arrow_dtype {
                    bail_unhandled_arrow_conversion_dtype_pair!(polars_dtype, &arrow_field)
                }

                if pl_dtype.is_logical() {
                    panic!("{pl_dtype:?}");
                }

                array.to_boxed()
            },
        })
    }

    /// Export a list of map entries as a `MapArray`.
    fn map_array_to_arrow(
        &mut self,
        array: &dyn Array,
        entries_dtype: &DataType,
        arrow_field: Cow<'_, ArrowField>,
    ) -> PolarsResult<Box<dyn Array>> {
        use polars_arrow::array::MapArray;
        use polars_arrow::offset::OffsetsBuffer;

        let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
        // Arrow's MAP entries and keys are non-nullable, and entries that no live row owns
        // may be null, so normalize before the child is read: those entries are dropped.
        let normalized = normalize_map_entries(arr);
        let arr = normalized.as_ref().unwrap_or(arr);

        let mut arrow_dtype = to_owned_dtype(arrow_field);

        let ArrowDataType::Map(arrow_entries_field, _keys_sorted) = &mut arrow_dtype else {
            unreachable!()
        };

        self.attach_pl_field_metadata(std::iter::once((
            entries_dtype,
            arrow_entries_field.as_mut(),
        )));

        let entries = self.array_to_arrow(
            arr.values().as_ref(),
            entries_dtype,
            Cow::Borrowed(arrow_entries_field.as_ref()),
        )?;

        // Arrow's MAP offsets are `i32`, a Polars list's are `i64`.
        let offsets = OffsetsBuffer::<i32>::try_from(arr.offsets()).map_err(|_| {
            polars_err!(
                InvalidOperation:
                "to_arrow() conversion failed: {} map entries overflow the i32 offsets \
                of the arrow MAP type",
                arr.offsets().last(),
            )
        })?;

        Ok(Box::new(MapArray::try_new(
            arrow_dtype,
            offsets,
            entries,
            arr.validity().cloned(),
        )?))
    }

    #[inline]
    fn attach_pl_field_metadata<'a, 'b, I>(&self, iter: I)
    where
        I: IntoIterator<Item = (&'a DataType, &'b mut ArrowField)>,
    {
        if self.skip_attach_pl_metadata {
            return;
        }

        inner(&mut iter.into_iter());

        #[inline(never)]
        fn inner(iter: &mut dyn Iterator<Item = (&DataType, &mut ArrowField)>) {
            for (pl_dtype, arrow_field) in iter {
                match pl_dtype {
                    #[cfg(feature = "dtype-categorical")]
                    DataType::Categorical(..) | DataType::Enum(..)
                        if !matches!(arrow_field.dtype(), ArrowDataType::Dictionary(..)) =>
                    {
                        // IPC sink can hit here when it exports only the keys of the categorical.
                        // In this case we do not want to attach categorical metadata.
                        continue;
                    },
                    _ => {},
                }

                let mut pl_md = pl_dtype.to_arrow_field_metadata();

                if arrow_field.metadata.is_none() {
                    arrow_field.metadata = pl_md.take().map(|x| x.into());
                }

                // Insert polars categorical and enum metadata.
                if let Some(pl_md) = pl_md
                    && let Some(md) = arrow_field.metadata.as_mut()
                {
                    for (k, v) in pl_md {
                        if !md.contains_key(&k) {
                            Arc::make_mut(md).insert(k, v);
                        }
                    }
                }
            }
        }
    }
}

fn to_owned_dtype(field: Cow<ArrowField>) -> ArrowDataType {
    match field {
        Cow::Borrowed(f) => f.dtype().clone(),
        Cow::Owned(f) => f.dtype,
    }
}

#[cfg(all(test, feature = "dtype-map"))]
mod tests {
    use polars_arrow::array::{BooleanArray, MapArray, NullArray};
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::offset::Offsets;

    use super::*;
    use crate::prelude::*;

    /// Rows that just fit in `i32` offsets two at a time.
    const ROW_LEN: i64 = (1 << 30) - 1;

    fn map_dtype() -> DataType {
        DataType::Map(Box::new(DataType::Boolean), Box::new(DataType::Null))
    }

    /// A `Map(Boolean, Null)` chunk with `row_lens` entries per row; the keys are a zeroed
    /// bitmap, so the entries cost no memory until read.
    fn map_chunk(row_lens: &[i64]) -> ArrayRef {
        let n = row_lens.iter().sum::<i64>() as usize;
        let entries_dtype = map_dtype().map_entries_dtype().unwrap();
        let keys = BooleanArray::new(ArrowDataType::Boolean, Bitmap::new_zeroed(n), None);
        let values = NullArray::new(ArrowDataType::Null, n);
        let entries = StructArray::new(
            entries_dtype.to_physical().to_arrow(CompatLevel::newest()),
            n,
            vec![keys.boxed(), values.boxed()],
            None,
        );
        list_chunk(row_lens, entries.boxed())
    }

    fn list_chunk(row_lens: &[i64], values: ArrayRef) -> ArrayRef {
        let offsets = Offsets::try_from_lengths(row_lens.iter().map(|&l| l as usize)).unwrap();
        ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(values.dtype().clone()),
            offsets.into(),
            values,
            None,
        )
        .boxed()
    }

    fn map_series(row_lens: &[i64]) -> Series {
        unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::from_static("m"),
                vec![map_chunk(row_lens)],
                &map_dtype(),
            )
        }
    }

    /// The entry counts of each row of an exported MAP array.
    fn map_row_lens(array: &dyn Array) -> Vec<i64> {
        let arr: &MapArray = array.as_any().downcast_ref().unwrap();
        assert_eq!(arr.offsets().range() as usize, arr.field().len());
        arr.offsets().lengths().map(|l| l as i64).collect()
    }

    #[test]
    fn iter_chunks_splits_a_map_chunk_that_overflows_i32_offsets() {
        let df = DataFrame::new_infer_height(vec![
            map_series(&[ROW_LEN; 4]).into_column(),
            Column::new("i".into(), [0i32, 1, 2, 3]),
        ])
        .unwrap();

        let batches = df
            .try_iter_chunks(CompatLevel::newest(), false)
            .collect::<PolarsResult<Vec<_>>>()
            .unwrap();

        assert_eq!(batches.len(), 2);
        for (batch, first) in batches.iter().zip([0i32, 2]) {
            assert_eq!(batch.len(), 2);
            assert_eq!(map_row_lens(batch.arrays()[0].as_ref()), [ROW_LEN; 2]);
            let ints: &PrimitiveArray<i32> = batch.arrays()[1].as_any().downcast_ref().unwrap();
            assert_eq!(ints.values().as_slice(), [first, first + 1]);
        }
    }

    #[test]
    fn try_to_arrow_split_splits_a_map_chunk_that_overflows_i32_offsets() {
        let arrays = map_series(&[ROW_LEN, ROW_LEN, 5, ROW_LEN])
            .try_to_arrow_split(0, CompatLevel::newest())
            .unwrap();

        let row_lens: Vec<_> = arrays.iter().map(|a| map_row_lens(a.as_ref())).collect();
        assert_eq!(row_lens, [vec![ROW_LEN, ROW_LEN], vec![5, ROW_LEN]]);
    }

    #[test]
    fn a_list_of_maps_is_split_on_the_nested_map_entries() {
        let lists = list_chunk(&[1; 4], map_chunk(&[ROW_LEN; 4]));
        let dtype = DataType::List(Box::new(map_dtype()));
        let s = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::from_static("l"),
                vec![lists],
                &dtype,
            )
        };

        let arrays = s.try_to_arrow_split(0, CompatLevel::newest()).unwrap();

        assert_eq!(arrays.len(), 2);
        for array in arrays {
            let list: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            assert_eq!(map_row_lens(list.values().as_ref()), [ROW_LEN; 2]);
        }
    }

    #[test]
    fn a_single_row_with_too_many_map_entries_errors() {
        let df =
            DataFrame::new_infer_height(vec![map_series(&[1, i32::MAX as i64 + 1]).into_column()])
                .unwrap();

        let err = df
            .try_iter_chunks(CompatLevel::newest(), false)
            .collect::<PolarsResult<Vec<_>>>()
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("a single row has 2147483648 map entries")
        );
    }
}
