// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines sort kernel for `ArrayRef`

use std::cmp::{Ordering, Reverse};

use crate::array::*;
use crate::compute::take;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};

use TimeUnit::*;

/// Sort the `ArrayRef` using `SortOptions`.
///
/// Performs a stable sort on values and indices, returning nulls after sorted valid values,
/// while preserving the order of the nulls.
///
/// Returns an `ArrowError::ComputeError(String)` if the array type is either unsupported by `sort_to_indices` or `take`.
pub fn sort(values: &ArrayRef, options: Option<SortOptions>) -> Result<ArrayRef> {
    let indices = sort_to_indices(values, options)?;
    take(values, &indices, None)
}

/// Sort elements from `ArrayRef` into an unsigned integer (`UInt32Array`) of indices
pub fn sort_to_indices(
    values: &ArrayRef,
    options: Option<SortOptions>,
) -> Result<UInt32Array> {
    let options = options.unwrap_or_default();
    let range = values.offset()..values.len();
    let (v, n): (Vec<usize>, Vec<usize>) =
        range.partition(|index| values.is_valid(*index));
    let n = n.into_iter().map(|i| i as u32).collect();
    match values.data_type() {
        DataType::Boolean => sort_primitive::<BooleanType>(values, v, n, &options),
        DataType::Int8 => sort_primitive::<Int8Type>(values, v, n, &options),
        DataType::Int16 => sort_primitive::<Int16Type>(values, v, n, &options),
        DataType::Int32 => sort_primitive::<Int32Type>(values, v, n, &options),
        DataType::Int64 => sort_primitive::<Int64Type>(values, v, n, &options),
        DataType::UInt8 => sort_primitive::<UInt8Type>(values, v, n, &options),
        DataType::UInt16 => sort_primitive::<UInt16Type>(values, v, n, &options),
        DataType::UInt32 => sort_primitive::<UInt32Type>(values, v, n, &options),
        DataType::UInt64 => sort_primitive::<UInt64Type>(values, v, n, &options),
        DataType::Date32(_) => sort_primitive::<Date32Type>(values, v, n, &options),
        DataType::Date64(_) => sort_primitive::<Date64Type>(values, v, n, &options),
        DataType::Time32(Second) => {
            sort_primitive::<Time32SecondType>(values, v, n, &options)
        }
        DataType::Time32(Millisecond) => {
            sort_primitive::<Time32MillisecondType>(values, v, n, &options)
        }
        DataType::Time64(Microsecond) => {
            sort_primitive::<Time64MicrosecondType>(values, v, n, &options)
        }
        DataType::Time64(Nanosecond) => {
            sort_primitive::<Time64NanosecondType>(values, v, n, &options)
        }
        DataType::Timestamp(Second, _) => {
            sort_primitive::<TimestampSecondType>(values, v, n, &options)
        }
        DataType::Timestamp(Millisecond, _) => {
            sort_primitive::<TimestampMillisecondType>(values, v, n, &options)
        }
        DataType::Timestamp(Microsecond, _) => {
            sort_primitive::<TimestampMicrosecondType>(values, v, n, &options)
        }
        DataType::Timestamp(Nanosecond, _) => {
            sort_primitive::<TimestampNanosecondType>(values, v, n, &options)
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            sort_primitive::<IntervalYearMonthType>(values, v, n, &options)
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            sort_primitive::<IntervalDayTimeType>(values, v, n, &options)
        }
        DataType::Duration(TimeUnit::Second) => {
            sort_primitive::<DurationSecondType>(values, v, n, &options)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            sort_primitive::<DurationMillisecondType>(values, v, n, &options)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            sort_primitive::<DurationMicrosecondType>(values, v, n, &options)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            sort_primitive::<DurationNanosecondType>(values, v, n, &options)
        }
        DataType::Utf8 => sort_string(values, v, n, &options),
        t => Err(ArrowError::ComputeError(format!(
            "Sort not supported for data type {:?}",
            t
        ))),
    }
}

/// Options that define how sort kernels should behave
#[derive(Clone, Copy)]
pub struct SortOptions {
    /// Whether to sort in descending order
    pub descending: bool,
    /// Whether to sort nulls first
    pub nulls_first: bool,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self {
            descending: false,
            // default to nulls first to match spark's behavior
            nulls_first: true,
        }
    }
}

/// Sort primitive values
fn sort_primitive<T>(
    values: &ArrayRef,
    value_indices: Vec<usize>,
    null_indices: Vec<u32>,
    options: &SortOptions,
) -> Result<UInt32Array>
where
    T: ArrowPrimitiveType,
    T::Native: std::cmp::Ord,
{
    let values = as_primitive_array::<T>(values);
    // create tuples that are used for sorting
    let mut valids = value_indices
        .into_iter()
        .map(|index| (index as u32, values.value(index)))
        .collect::<Vec<(u32, T::Native)>>();
    let mut nulls = null_indices;
    if !options.descending {
        valids.sort_by_key(|a| a.1);
    } else {
        valids.sort_by_key(|a| Reverse(a.1));
        nulls.reverse();
    }
    // collect the order of valid tuples
    let mut valid_indices: Vec<u32> = valids.iter().map(|tuple| tuple.0).collect();

    if options.nulls_first {
        nulls.append(&mut valid_indices);
        return Ok(UInt32Array::from(nulls));
    }
    // no need to sort nulls as they are in the correct order already
    valid_indices.append(&mut nulls);

    Ok(UInt32Array::from(valid_indices))
}

/// Sort strings
fn sort_string(
    values: &ArrayRef,
    value_indices: Vec<usize>,
    null_indices: Vec<u32>,
    options: &SortOptions,
) -> Result<UInt32Array> {
    let values = as_string_array(values);
    let mut valids = value_indices
        .into_iter()
        .map(|index| (index as u32, values.value(index)))
        .collect::<Vec<(u32, &str)>>();
    let mut nulls = null_indices;
    if !options.descending {
        valids.sort_by_key(|a| a.1);
    } else {
        valids.sort_by_key(|a| Reverse(a.1));
        nulls.reverse();
    }
    // collect the order of valid tuplies
    let mut valid_indices: Vec<u32> = valids.iter().map(|tuple| tuple.0).collect();

    if options.nulls_first {
        nulls.append(&mut valid_indices);
        return Ok(UInt32Array::from(nulls));
    }

    // no need to sort nulls as they are in the correct order already
    valid_indices.append(&mut nulls);

    Ok(UInt32Array::from(valid_indices))
}

/// One column to be used in lexicographical sort
#[derive(Clone)]
pub struct SortColumn {
    pub values: ArrayRef,
    pub options: Option<SortOptions>,
}

/// Sort a list of `ArrayRef` using `SortOptions` provided for each array.
///
/// Performs a stable lexicographical sort on values and indices.
///
/// Returns an `ArrowError::ComputeError(String)` if any of the array type is either unsupported by
/// `lexsort_to_indices` or `take`.
///
/// Example:
///
/// ```
/// use std::convert::TryFrom;
/// use std::sync::Arc;
/// use some::array::{ArrayRef, StringArray, PrimitiveArray, as_primitive_array};
/// use some::compute::kernels::sort::{SortColumn, SortOptions, lexsort};
/// use some::datatypes::Int64Type;
///
/// let sorted_columns = lexsort(&vec![
///     SortColumn {
///         values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
///             None,
///             Some(-2),
///             Some(89),
///             Some(-64),
///             Some(101),
///         ])) as ArrayRef,
///         options: None,
///     },
///     SortColumn {
///         values: Arc::new(StringArray::try_from(vec![
///             Some("hello"),
///             Some("world"),
///             Some(","),
///             Some("foobar"),
///             Some("!"),
///         ]).unwrap()) as ArrayRef,
///         options: Some(SortOptions {
///             descending: true,
///             nulls_first: false,
///         }),
///     },
/// ]).unwrap();
///
/// assert_eq!(as_primitive_array::<Int64Type>(&sorted_columns[0]).value(1), -64);
/// assert!(sorted_columns[0].is_null(0));
/// ```
pub fn lexsort(columns: &Vec<SortColumn>) -> Result<Vec<ArrayRef>> {
    let indices = lexsort_to_indices(columns)?;
    columns
        .iter()
        .map(|c| take(&c.values, &indices, None))
        .collect()
}

/// Sort elements lexicographically from a list of `ArrayRef` into an unsigned integer
/// (`UInt32Array`) of indices.
pub fn lexsort_to_indices(columns: &Vec<SortColumn>) -> Result<UInt32Array> {
    if columns.len() == 1 {
        // fallback to non-lexical sort
        let column = &columns[0];
        return sort_to_indices(&column.values, column.options);
    }

    let mut row_count = None;
    // convert ArrayRefs to OrdArray trait objects and perform row count check
    let flat_columns = columns
        .iter()
        .map(|column| -> Result<(Box<&OrdArray>, SortOptions)> {
            // row count check
            let curr_row_count = column.values.len() - column.values.offset();
            match row_count {
                None => {
                    row_count = Some(curr_row_count);
                }
                Some(cnt) => {
                    if curr_row_count != cnt {
                        return Err(ArrowError::ComputeError(
                            "lexical sort columns have different row counts".to_string(),
                        ));
                    }
                }
            }
            // flatten and convert to OrdArray
            Ok((
                as_ordarray(&column.values)?,
                column.options.unwrap_or_default(),
            ))
        })
        .collect::<Result<Vec<(Box<&OrdArray>, SortOptions)>>>()?;

    let lex_comparator = |a_idx: &usize, b_idx: &usize| -> Ordering {
        for column in flat_columns.iter() {
            let values = &column.0;
            let sort_option = column.1;

            match (values.is_valid(*a_idx), values.is_valid(*b_idx)) {
                (true, true) => {
                    match values.cmp_value(*a_idx, *b_idx) {
                        // equal, move on to next column
                        Ordering::Equal => continue,
                        order @ _ => {
                            if sort_option.descending {
                                return order.reverse();
                            } else {
                                return order;
                            }
                        }
                    }
                }
                (false, true) => {
                    return if sort_option.nulls_first {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                }
                (true, false) => {
                    return if sort_option.nulls_first {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };
                }
                // equal, move on to next column
                (false, false) => continue,
            }
        }

        Ordering::Equal
    };

    let mut value_indices = (0..row_count.unwrap()).collect::<Vec<usize>>();
    value_indices.sort_by(lex_comparator);

    Ok(UInt32Array::from(
        value_indices
            .into_iter()
            .map(|i| i as u32)
            .collect::<Vec<u32>>(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{convert::TryFrom, sync::Arc};

    fn test_sort_to_indices_primitive_arrays<T>(
        data: Vec<Option<T::Native>>,
        options: Option<SortOptions>,
        expected_data: Vec<u32>,
    ) where
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>> + ArrayEqual,
    {
        let output = PrimitiveArray::<T>::from(data);
        let expected = UInt32Array::from(expected_data);
        let output = sort_to_indices(&(Arc::new(output) as ArrayRef), options).unwrap();
        assert!(output.equals(&expected))
    }

    fn test_sort_primitive_arrays<T>(
        data: Vec<Option<T::Native>>,
        options: Option<SortOptions>,
        expected_data: Vec<Option<T::Native>>,
    ) where
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>> + ArrayEqual,
    {
        let output = PrimitiveArray::<T>::from(data);
        let expected = PrimitiveArray::<T>::from(expected_data);
        let output = sort(&(Arc::new(output) as ArrayRef), options).unwrap();
        let output = output.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        assert!(output.equals(&expected))
    }

    fn test_sort_to_indices_string_arrays(
        data: Vec<Option<&str>>,
        options: Option<SortOptions>,
        expected_data: Vec<u32>,
    ) {
        let output = StringArray::try_from(data).expect("Unable to create string array");
        let expected = UInt32Array::from(expected_data);
        let output = sort_to_indices(&(Arc::new(output) as ArrayRef), options).unwrap();
        assert!(output.equals(&expected))
    }

    fn test_sort_string_arrays(
        data: Vec<Option<&str>>,
        options: Option<SortOptions>,
        expected_data: Vec<Option<&str>>,
    ) {
        let output = StringArray::try_from(data).expect("Unable to create string array");
        let expected =
            StringArray::try_from(expected_data).expect("Unable to create string array");
        let output = sort(&(Arc::new(output) as ArrayRef), options).unwrap();
        let output = output.as_any().downcast_ref::<StringArray>().unwrap();
        assert!(output.equals(&expected))
    }

    fn test_lex_sort_arrays(input: Vec<SortColumn>, expected_output: Vec<ArrayRef>) {
        let sorted = lexsort(&input).unwrap();
        let sorted2cmp = sorted.iter().map(|arr| -> Box<&dyn ArrayEqual> {
            match arr.data_type() {
                DataType::Int64 => Box::new(as_primitive_array::<Int64Type>(&arr)),
                DataType::UInt32 => Box::new(as_primitive_array::<UInt32Type>(&arr)),
                DataType::Utf8 => Box::new(as_string_array(&arr)),
                _ => panic!("unexpected array type"),
            }
        });
        for (i, values) in sorted2cmp.enumerate() {
            assert!(
                values.equals(&(*expected_output[i])),
                "expect {:#?} to be: {:#?}",
                sorted,
                expected_output
            );
        }
    }

    #[test]
    fn test_sort_to_indices_primitives() {
        test_sort_to_indices_primitive_arrays::<Int8Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            None,
            vec![0, 5, 3, 1, 4, 2],
        );
        test_sort_to_indices_primitive_arrays::<Int16Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            None,
            vec![0, 5, 3, 1, 4, 2],
        );
        test_sort_to_indices_primitive_arrays::<Int32Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            None,
            vec![0, 5, 3, 1, 4, 2],
        );
        test_sort_to_indices_primitive_arrays::<Int64Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            None,
            vec![0, 5, 3, 1, 4, 2],
        );

        // descending
        test_sort_to_indices_primitive_arrays::<Int8Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![2, 1, 4, 3, 5, 0], // [2, 4, 1, 3, 5, 0]
        );

        test_sort_to_indices_primitive_arrays::<Int16Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![2, 1, 4, 3, 5, 0],
        );

        test_sort_to_indices_primitive_arrays::<Int32Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![2, 1, 4, 3, 5, 0],
        );

        test_sort_to_indices_primitive_arrays::<Int64Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![2, 1, 4, 3, 5, 0],
        );

        // descending, nulls first
        test_sort_to_indices_primitive_arrays::<Int8Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![5, 0, 2, 1, 4, 3], // [5, 0, 2, 4, 1, 3]
        );

        test_sort_to_indices_primitive_arrays::<Int16Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![5, 0, 2, 1, 4, 3], // [5, 0, 2, 4, 1, 3]
        );

        test_sort_to_indices_primitive_arrays::<Int32Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![5, 0, 2, 1, 4, 3],
        );

        test_sort_to_indices_primitive_arrays::<Int64Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![5, 0, 2, 1, 4, 3],
        );

        // boolean
        test_sort_to_indices_primitive_arrays::<BooleanType>(
            vec![None, Some(false), Some(true), Some(true), Some(false), None],
            None,
            vec![0, 5, 1, 4, 2, 3],
        );

        // boolean, descending
        test_sort_to_indices_primitive_arrays::<BooleanType>(
            vec![None, Some(false), Some(true), Some(true), Some(false), None],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![2, 3, 1, 4, 5, 0],
        );

        // boolean, descending, nulls first
        test_sort_to_indices_primitive_arrays::<BooleanType>(
            vec![None, Some(false), Some(true), Some(true), Some(false), None],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![5, 0, 2, 3, 1, 4],
        );
    }

    #[test]
    fn test_sort_primitives() {
        // default case
        test_sort_primitive_arrays::<UInt8Type>(
            vec![None, Some(3), Some(5), Some(2), Some(3), None],
            None,
            vec![None, None, Some(2), Some(3), Some(3), Some(5)],
        );
        test_sort_primitive_arrays::<UInt16Type>(
            vec![None, Some(3), Some(5), Some(2), Some(3), None],
            None,
            vec![None, None, Some(2), Some(3), Some(3), Some(5)],
        );
        test_sort_primitive_arrays::<UInt32Type>(
            vec![None, Some(3), Some(5), Some(2), Some(3), None],
            None,
            vec![None, None, Some(2), Some(3), Some(3), Some(5)],
        );
        test_sort_primitive_arrays::<UInt64Type>(
            vec![None, Some(3), Some(5), Some(2), Some(3), None],
            None,
            vec![None, None, Some(2), Some(3), Some(3), Some(5)],
        );

        // descending
        test_sort_primitive_arrays::<Int8Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![Some(2), Some(0), Some(0), Some(-1), None, None],
        );
        test_sort_primitive_arrays::<Int16Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![Some(2), Some(0), Some(0), Some(-1), None, None],
        );
        test_sort_primitive_arrays::<Int32Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![Some(2), Some(0), Some(0), Some(-1), None, None],
        );
        test_sort_primitive_arrays::<Int16Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![Some(2), Some(0), Some(0), Some(-1), None, None],
        );

        // descending, nulls first
        test_sort_primitive_arrays::<Int8Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![None, None, Some(2), Some(0), Some(0), Some(-1)],
        );
        test_sort_primitive_arrays::<Int16Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![None, None, Some(2), Some(0), Some(0), Some(-1)],
        );
        test_sort_primitive_arrays::<Int32Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![None, None, Some(2), Some(0), Some(0), Some(-1)],
        );
        test_sort_primitive_arrays::<Int64Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![None, None, Some(2), Some(0), Some(0), Some(-1)],
        );

        // int8 nulls first
        test_sort_primitive_arrays::<Int8Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
            vec![None, None, Some(-1), Some(0), Some(0), Some(2)],
        );
        test_sort_primitive_arrays::<Int16Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
            vec![None, None, Some(-1), Some(0), Some(0), Some(2)],
        );
        test_sort_primitive_arrays::<Int32Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
            vec![None, None, Some(-1), Some(0), Some(0), Some(2)],
        );
        test_sort_primitive_arrays::<Int64Type>(
            vec![None, Some(0), Some(2), Some(-1), Some(0), None],
            Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
            vec![None, None, Some(-1), Some(0), Some(0), Some(2)],
        );
    }

    #[test]
    fn test_sort_to_indices_strings() {
        test_sort_to_indices_string_arrays(
            vec![
                None,
                Some("bad"),
                Some("sad"),
                None,
                Some("glad"),
                Some("-ad"),
            ],
            None,
            vec![0, 3, 5, 1, 4, 2],
        );

        test_sort_to_indices_string_arrays(
            vec![
                None,
                Some("bad"),
                Some("sad"),
                None,
                Some("glad"),
                Some("-ad"),
            ],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![2, 4, 1, 5, 3, 0],
        );

        test_sort_to_indices_string_arrays(
            vec![
                None,
                Some("bad"),
                Some("sad"),
                None,
                Some("glad"),
                Some("-ad"),
            ],
            Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
            vec![0, 3, 5, 1, 4, 2],
        );

        test_sort_to_indices_string_arrays(
            vec![
                None,
                Some("bad"),
                Some("sad"),
                None,
                Some("glad"),
                Some("-ad"),
            ],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![3, 0, 2, 4, 1, 5],
        );
    }

    #[test]
    fn test_sort_strings() {
        test_sort_string_arrays(
            vec![
                None,
                Some("bad"),
                Some("sad"),
                None,
                Some("glad"),
                Some("-ad"),
            ],
            None,
            vec![
                None,
                None,
                Some("-ad"),
                Some("bad"),
                Some("glad"),
                Some("sad"),
            ],
        );

        test_sort_string_arrays(
            vec![
                None,
                Some("bad"),
                Some("sad"),
                None,
                Some("glad"),
                Some("-ad"),
            ],
            Some(SortOptions {
                descending: true,
                nulls_first: false,
            }),
            vec![
                Some("sad"),
                Some("glad"),
                Some("bad"),
                Some("-ad"),
                None,
                None,
            ],
        );

        test_sort_string_arrays(
            vec![
                None,
                Some("bad"),
                Some("sad"),
                None,
                Some("glad"),
                Some("-ad"),
            ],
            Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
            vec![
                None,
                None,
                Some("-ad"),
                Some("bad"),
                Some("glad"),
                Some("sad"),
            ],
        );

        test_sort_string_arrays(
            vec![
                None,
                Some("bad"),
                Some("sad"),
                None,
                Some("glad"),
                Some("-ad"),
            ],
            Some(SortOptions {
                descending: true,
                nulls_first: true,
            }),
            vec![
                None,
                None,
                Some("sad"),
                Some("glad"),
                Some("bad"),
                Some("-ad"),
            ],
        );
    }

    #[test]
    fn test_lex_sort_single_column() {
        let input = vec![SortColumn {
            values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(17),
                Some(2),
                Some(-1),
                Some(0),
            ])) as ArrayRef,
            options: None,
        }];
        let expected = vec![Arc::new(PrimitiveArray::<Int64Type>::from(vec![
            Some(-1),
            Some(0),
            Some(2),
            Some(17),
        ])) as ArrayRef];
        test_lex_sort_arrays(input, expected);
    }

    #[test]
    fn test_lex_sort_unaligned_rows() {
        let input = vec![
            SortColumn {
                values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![None, Some(-1)]))
                    as ArrayRef,
                options: None,
            },
            SortColumn {
                values: Arc::new(
                    StringArray::try_from(vec![Some("foo")])
                        .expect("Unable to create string array"),
                ) as ArrayRef,
                options: None,
            },
        ];
        assert!(
            lexsort(&input).is_err(),
            "lexsort should reject columns with different row counts"
        );
    }

    #[test]
    fn test_lex_sort_mixed_types() {
        let input = vec![
            SortColumn {
                values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                    Some(0),
                    Some(2),
                    Some(-1),
                    Some(0),
                ])) as ArrayRef,
                options: None,
            },
            SortColumn {
                values: Arc::new(PrimitiveArray::<UInt32Type>::from(vec![
                    Some(101),
                    Some(8),
                    Some(7),
                    Some(102),
                ])) as ArrayRef,
                options: None,
            },
            SortColumn {
                values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                    Some(-1),
                    Some(-2),
                    Some(-3),
                    Some(-4),
                ])) as ArrayRef,
                options: None,
            },
        ];
        let expected = vec![
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(-1),
                Some(0),
                Some(0),
                Some(2),
            ])) as ArrayRef,
            Arc::new(PrimitiveArray::<UInt32Type>::from(vec![
                Some(7),
                Some(101),
                Some(102),
                Some(8),
            ])) as ArrayRef,
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(-3),
                Some(-1),
                Some(-4),
                Some(-2),
            ])) as ArrayRef,
        ];
        test_lex_sort_arrays(input, expected);

        // test mix of string and in64 with option
        let input = vec![
            SortColumn {
                values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                    Some(0),
                    Some(2),
                    Some(-1),
                    Some(0),
                ])) as ArrayRef,
                options: Some(SortOptions {
                    descending: true,
                    nulls_first: true,
                }),
            },
            SortColumn {
                values: Arc::new(
                    StringArray::try_from(vec![
                        Some("foo"),
                        Some("9"),
                        Some("7"),
                        Some("bar"),
                    ])
                    .expect("Unable to create string array"),
                ) as ArrayRef,
                options: Some(SortOptions {
                    descending: true,
                    nulls_first: true,
                }),
            },
        ];
        let expected = vec![
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(2),
                Some(0),
                Some(0),
                Some(-1),
            ])) as ArrayRef,
            Arc::new(
                StringArray::try_from(vec![
                    Some("9"),
                    Some("foo"),
                    Some("bar"),
                    Some("7"),
                ])
                .expect("Unable to create string array"),
            ) as ArrayRef,
        ];
        test_lex_sort_arrays(input, expected);

        // test sort with nulls first
        let input = vec![
            SortColumn {
                values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                    None,
                    Some(-1),
                    Some(2),
                    None,
                ])) as ArrayRef,
                options: Some(SortOptions {
                    descending: true,
                    nulls_first: true,
                }),
            },
            SortColumn {
                values: Arc::new(
                    StringArray::try_from(vec![
                        Some("foo"),
                        Some("world"),
                        Some("hello"),
                        None,
                    ])
                    .expect("Unable to create string array"),
                ) as ArrayRef,
                options: Some(SortOptions {
                    descending: true,
                    nulls_first: true,
                }),
            },
        ];
        let expected = vec![
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                None,
                None,
                Some(2),
                Some(-1),
            ])) as ArrayRef,
            Arc::new(
                StringArray::try_from(vec![
                    None,
                    Some("foo"),
                    Some("hello"),
                    Some("world"),
                ])
                .expect("Unable to create string array"),
            ) as ArrayRef,
        ];
        test_lex_sort_arrays(input, expected);

        // test sort with nulls last
        let input = vec![
            SortColumn {
                values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                    None,
                    Some(-1),
                    Some(2),
                    None,
                ])) as ArrayRef,
                options: Some(SortOptions {
                    descending: true,
                    nulls_first: false,
                }),
            },
            SortColumn {
                values: Arc::new(
                    StringArray::try_from(vec![
                        Some("foo"),
                        Some("world"),
                        Some("hello"),
                        None,
                    ])
                    .expect("Unable to create string array"),
                ) as ArrayRef,
                options: Some(SortOptions {
                    descending: true,
                    nulls_first: false,
                }),
            },
        ];
        let expected = vec![
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(2),
                Some(-1),
                None,
                None,
            ])) as ArrayRef,
            Arc::new(
                StringArray::try_from(vec![
                    Some("hello"),
                    Some("world"),
                    Some("foo"),
                    None,
                ])
                .expect("Unable to create string array"),
            ) as ArrayRef,
        ];
        test_lex_sort_arrays(input, expected);

        // test sort with opposite options
        let input = vec![
            SortColumn {
                values: Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                    None,
                    Some(-1),
                    Some(2),
                    Some(-1),
                    None,
                ])) as ArrayRef,
                options: Some(SortOptions {
                    descending: false,
                    nulls_first: false,
                }),
            },
            SortColumn {
                values: Arc::new(
                    StringArray::try_from(vec![
                        Some("foo"),
                        Some("bar"),
                        Some("world"),
                        Some("hello"),
                        None,
                    ])
                    .expect("Unable to create string array"),
                ) as ArrayRef,
                options: Some(SortOptions {
                    descending: true,
                    nulls_first: true,
                }),
            },
        ];
        let expected = vec![
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(-1),
                Some(-1),
                Some(2),
                None,
                None,
            ])) as ArrayRef,
            Arc::new(
                StringArray::try_from(vec![
                    Some("hello"),
                    Some("bar"),
                    Some("world"),
                    None,
                    Some("foo"),
                ])
                .expect("Unable to create string array"),
            ) as ArrayRef,
        ];
        test_lex_sort_arrays(input, expected);
    }
}
