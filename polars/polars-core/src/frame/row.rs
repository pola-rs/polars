use std::borrow::Borrow;
use std::fmt::Debug;

use arrow::bitmap::Bitmap;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::try_get_supertype;
use crate::POOL;
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Row<'a>(pub Vec<AnyValue<'a>>);

impl<'a> Row<'a> {
    pub fn new(values: Vec<AnyValue<'a>>) -> Self {
        Row(values)
    }
}

impl DataFrame {
    /// Get a row from a DataFrame. Use of this is discouraged as it will likely be slow.
    pub fn get_row(&self, idx: usize) -> PolarsResult<Row> {
        let values = self
            .columns
            .iter()
            .map(|s| s.get(idx))
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(Row(values))
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible to make sure that the row has at least the capacity for the number
    /// of columns in the DataFrame
    pub fn get_row_amortized<'a>(&'a self, idx: usize, row: &mut Row<'a>) -> PolarsResult<()> {
        for (s, any_val) in self.columns.iter().zip(&mut row.0) {
            *any_val = s.get(idx)?;
        }
        Ok(())
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible to make sure that the row has at least the capacity for the number
    /// of columns in the DataFrame
    ///
    /// # Safety
    /// Does not do any bounds checking.
    #[inline]
    pub unsafe fn get_row_amortized_unchecked<'a>(&'a self, idx: usize, row: &mut Row<'a>) {
        self.columns
            .iter()
            .zip(&mut row.0)
            .for_each(|(s, any_val)| {
                *any_val = s.get_unchecked(idx);
            });
    }

    /// Create a new DataFrame from rows. This should only be used when you have row wise data,
    /// as this is a lot slower than creating the `Series` in a columnar fashion
    pub fn from_rows_and_schema(rows: &[Row], schema: &Schema) -> PolarsResult<Self> {
        Self::from_rows_iter_and_schema(rows.iter(), schema)
    }

    /// Create a new DataFrame from an iterator over rows. This should only be used when you have row wise data,
    /// as this is a lot slower than creating the `Series` in a columnar fashion
    pub fn from_rows_iter_and_schema<'a, I>(mut rows: I, schema: &Schema) -> PolarsResult<Self>
    where
        I: Iterator<Item = &'a Row<'a>>,
    {
        let capacity = rows.size_hint().0;

        let mut buffers: Vec<_> = schema
            .iter_dtypes()
            .map(|dtype| {
                let buf: AnyValueBuffer = (dtype, capacity).into();
                buf
            })
            .collect();

        let mut expected_len = 0;
        rows.try_for_each::<_, PolarsResult<()>>(|row| {
            expected_len += 1;
            for (value, buf) in row.0.iter().zip(&mut buffers) {
                buf.add_fallible(value)?
            }
            Ok(())
        })?;
        let v = buffers
            .into_iter()
            .zip(schema.iter_names())
            .map(|(b, name)| {
                let mut s = b.into_series();
                // if the schema adds a column not in the rows, we
                // fill it with nulls
                if s.is_empty() {
                    Series::full_null(name, expected_len, s.dtype())
                } else {
                    s.rename(name);
                    s
                }
            })
            .collect();
        DataFrame::new(v)
    }

    /// Create a new DataFrame from rows. This should only be used when you have row wise data,
    /// as this is a lot slower than creating the `Series` in a columnar fashion
    pub fn from_rows(rows: &[Row]) -> PolarsResult<Self> {
        let schema = rows_to_schema_first_non_null(rows, Some(50));
        let has_nulls = schema
            .iter_dtypes()
            .any(|dtype| matches!(dtype, DataType::Null));
        if has_nulls {
            return Err(PolarsError::ComputeError(
                "Could not infer row types, because of the null values".into(),
            ));
        }
        Self::from_rows_and_schema(rows, &schema)
    }

    pub(crate) fn transpose_from_dtype(&self, dtype: &DataType) -> PolarsResult<DataFrame> {
        let new_width = self.height();
        let new_height = self.width();

        match dtype {
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => numeric_transpose::<Int8Type>(&self.columns),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => numeric_transpose::<Int16Type>(&self.columns),
            DataType::Int32 => numeric_transpose::<Int32Type>(&self.columns),
            DataType::Int64 => numeric_transpose::<Int64Type>(&self.columns),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => numeric_transpose::<UInt8Type>(&self.columns),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => numeric_transpose::<UInt16Type>(&self.columns),
            DataType::UInt32 => numeric_transpose::<UInt32Type>(&self.columns),
            DataType::UInt64 => numeric_transpose::<UInt64Type>(&self.columns),
            DataType::Float32 => numeric_transpose::<Float32Type>(&self.columns),
            DataType::Float64 => numeric_transpose::<Float64Type>(&self.columns),
            _ => {
                let mut buffers = (0..new_width)
                    .map(|_| {
                        let buf: AnyValueBuffer = (dtype, new_height).into();
                        buf
                    })
                    .collect::<Vec<_>>();

                let columns = self
                    .columns
                    .iter()
                    .map(|s| s.cast(dtype).unwrap())
                    .collect::<Vec<_>>();

                // this is very expensive. A lot of cache misses here.
                // This is the part that is performance critical.
                columns.iter().for_each(|s| {
                    s.iter().zip(buffers.iter_mut()).for_each(|(av, buf)| {
                        let _out = buf.add(av);
                        debug_assert!(_out.is_some());
                    });
                });
                let cols = buffers
                    .into_iter()
                    .enumerate()
                    .map(|(i, buf)| {
                        let mut s = buf.into_series();
                        s.rename(&format!("column_{i}"));
                        s
                    })
                    .collect::<Vec<_>>();
                Ok(DataFrame::new_no_checks(cols))
            }
        }
    }

    /// Transpose a DataFrame. This is a very expensive operation.
    pub fn transpose(&self) -> PolarsResult<DataFrame> {
        let height = self.height();
        let width = self.width();
        if height == 0 || width == 0 {
            return Err(PolarsError::NoData("empty dataframe".into()));
        }

        let dtype = self.get_supertype().unwrap()?;
        self.transpose_from_dtype(&dtype)
    }
}

type Tracker = PlIndexMap<String, PlHashSet<DataType>>;

pub fn infer_schema(
    iter: impl Iterator<Item = Vec<(String, impl Into<DataType>)>>,
    infer_schema_length: usize,
) -> Schema {
    let mut values: Tracker = Tracker::default();
    let len = iter.size_hint().1.unwrap_or(infer_schema_length);

    let max_infer = std::cmp::min(len, infer_schema_length);
    for inner in iter.take(max_infer) {
        for (key, value) in inner {
            add_or_insert(&mut values, &key, value.into());
        }
    }
    Schema::from(resolve_fields(values).into_iter())
}

fn add_or_insert(values: &mut Tracker, key: &str, data_type: DataType) {
    if data_type == DataType::Null {
        return;
    }

    if values.contains_key(key) {
        let x = values.get_mut(key).unwrap();
        x.insert(data_type);
    } else {
        // create hashset and add value type
        let mut hs = PlHashSet::new();
        hs.insert(data_type);
        values.insert(key.to_string(), hs);
    }
}

fn resolve_fields(spec: Tracker) -> Vec<Field> {
    spec.iter()
        .map(|(k, hs)| {
            let v: Vec<&DataType> = hs.iter().collect();
            Field::new(k, coerce_data_type(&v))
        })
        .collect()
}

/// Coerces a slice of datatypes into a single supertype.
pub fn coerce_data_type<A: Borrow<DataType>>(datatypes: &[A]) -> DataType {
    use DataType::*;

    let are_all_equal = datatypes.windows(2).all(|w| w[0].borrow() == w[1].borrow());

    if are_all_equal {
        return datatypes[0].borrow().clone();
    }
    if datatypes.len() > 2 {
        return Utf8;
    }

    let (lhs, rhs) = (datatypes[0].borrow(), datatypes[1].borrow());
    try_get_supertype(lhs, rhs).unwrap_or(Utf8)
}

fn is_nested_null(av: &AnyValue) -> bool {
    match av {
        AnyValue::Null => true,
        AnyValue::List(s) => s.null_count() == s.len(),
        #[cfg(feature = "dtype-struct")]
        AnyValue::Struct(_, _, _) => av._iter_struct_av().all(|av| is_nested_null(&av)),
        _ => false,
    }
}

// nested dtypes that are all null, will be set as null leave dtype
fn infer_dtype_dynamic(av: &AnyValue) -> DataType {
    match av {
        AnyValue::List(s) if s.null_count() == s.len() => DataType::List(Box::new(DataType::Null)),
        #[cfg(feature = "dtype-struct")]
        AnyValue::Struct(_, _, _) => DataType::Struct(
            av._iter_struct_av()
                .map(|av| {
                    let dtype = infer_dtype_dynamic(&av);
                    Field::new("", dtype)
                })
                .collect(),
        ),
        av => av.into(),
    }
}

pub fn any_values_to_dtype(column: &[AnyValue]) -> PolarsResult<DataType> {
    // we need an index-map as the order of dtypes influences how the
    // struct fields are constructed.
    let mut types_set = PlIndexSet::new();
    for val in column.iter() {
        let dtype = infer_dtype_dynamic(val);
        types_set.insert(dtype);
    }
    types_set_to_dtype(types_set)
}

fn types_set_to_dtype(types_set: PlIndexSet<DataType>) -> PolarsResult<DataType> {
    types_set
        .into_iter()
        .map(Ok)
        .fold_first_(|a, b| try_get_supertype(&a?, &b?))
        .unwrap()
}

/// Infer schema from rows and set the supertypes of the columns as column data type.
pub fn rows_to_schema_supertypes(
    rows: &[Row],
    infer_schema_length: Option<usize>,
) -> PolarsResult<Schema> {
    // no of rows to use to infer dtype
    let max_infer = infer_schema_length.unwrap_or(rows.len());

    if rows.is_empty() {
        return PolarsResult::Err(PolarsError::NoData("No rows. Cannot infer schema.".into()));
    }
    let mut dtypes: Vec<PlIndexSet<DataType>> = vec![PlIndexSet::new(); rows[0].0.len()];

    for row in rows.iter().take(max_infer) {
        for (val, types_set) in row.0.iter().zip(dtypes.iter_mut()) {
            let dtype = infer_dtype_dynamic(val);
            types_set.insert(dtype);
        }
    }

    dtypes
        .into_iter()
        .enumerate()
        .map(|(i, types_set)| {
            let dtype = if types_set.is_empty() {
                DataType::Unknown
            } else {
                types_set_to_dtype(types_set)?
            };
            Ok(Field::new(format!("column_{i}").as_ref(), dtype))
        })
        .collect::<PolarsResult<_>>()
}

/// Infer schema from rows and set the first no null type as column data type.
pub fn rows_to_schema_first_non_null(rows: &[Row], infer_schema_length: Option<usize>) -> Schema {
    // no of rows to use to infer dtype
    let max_infer = infer_schema_length.unwrap_or(rows.len());
    let mut schema: Schema = (&rows[0]).into();

    // the first row that has no nulls will be used to infer the schema.
    // if there is a null, we check the next row and see if we can update the schema

    for row in rows.iter().take(max_infer).skip(1) {
        // for i in 1..max_infer {
        let nulls: Vec<_> = schema
            .iter_dtypes()
            .enumerate()
            .filter_map(|(i, dtype)| {
                // double check struct and list types types
                // nested null values can be wrongly inferred by front ends
                match dtype {
                    DataType::Null | DataType::List(_) => Some(i),
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => Some(i),
                    _ => None,
                }
            })
            .collect();
        if nulls.is_empty() {
            break;
        } else {
            for i in nulls {
                let val = &row.0[i];

                if !is_nested_null(val) {
                    let dtype = val.into();
                    schema.coerce_by_index(i, dtype).unwrap();
                }
            }
        }
    }
    schema
}

impl<'a> From<&AnyValue<'a>> for Field {
    fn from(val: &AnyValue<'a>) -> Self {
        Field::new("", val.into())
    }
}

impl From<&Row<'_>> for Schema {
    fn from(row: &Row) -> Self {
        let fields = row.0.iter().enumerate().map(|(i, av)| {
            let dtype = av.into();
            Field::new(format!("column_{i}").as_ref(), dtype)
        });

        Schema::from(fields)
    }
}

pub enum AnyValueBuffer<'a> {
    Boolean(BooleanChunkedBuilder),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    #[cfg(feature = "dtype-date")]
    Date(PrimitiveChunkedBuilder<Int32Type>),
    #[cfg(feature = "dtype-datetime")]
    Datetime(
        PrimitiveChunkedBuilder<Int64Type>,
        TimeUnit,
        Option<TimeZone>,
    ),
    #[cfg(feature = "dtype-duration")]
    Duration(PrimitiveChunkedBuilder<Int64Type>, TimeUnit),
    #[cfg(feature = "dtype-time")]
    Time(PrimitiveChunkedBuilder<Int64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    Utf8(Utf8ChunkedBuilder),
    All(DataType, Vec<AnyValue<'a>>),
}

impl<'a> AnyValueBuffer<'a> {
    #[inline]
    pub fn add(&mut self, val: AnyValue<'a>) -> Option<()> {
        use AnyValueBuffer::*;
        match (self, val) {
            (Boolean(builder), AnyValue::Null) => builder.append_null(),
            (Boolean(builder), AnyValue::Boolean(v)) => builder.append_value(v),
            (Boolean(builder), val) => {
                let v = val.extract::<u8>()?;
                builder.append_value(v == 1)
            }
            (Int32(builder), AnyValue::Null) => builder.append_null(),
            (Int32(builder), val) => builder.append_value(val.extract()?),
            (Int64(builder), AnyValue::Null) => builder.append_null(),
            (Int64(builder), val) => builder.append_value(val.extract()?),
            (UInt32(builder), AnyValue::Null) => builder.append_null(),
            (UInt32(builder), val) => builder.append_value(val.extract()?),
            (UInt64(builder), AnyValue::Null) => builder.append_null(),
            (UInt64(builder), val) => builder.append_value(val.extract()?),
            #[cfg(feature = "dtype-date")]
            (Date(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-date")]
            (Date(builder), AnyValue::Date(v)) => builder.append_value(v),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(builder, _, _), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(builder, tu_l, _), AnyValue::Datetime(v, tu_r, _)) => {
                // we convert right tu to left tu
                // so we swap.
                let v = convert_time_units(v, tu_r, *tu_l);
                builder.append_value(v)
            }
            #[cfg(feature = "dtype-duration")]
            (Duration(builder, _), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-duration")]
            (Duration(builder, tu_l), AnyValue::Duration(v, tu_r)) => {
                let v = convert_time_units(v, tu_r, *tu_l);
                builder.append_value(v)
            }
            #[cfg(feature = "dtype-time")]
            (Time(builder), AnyValue::Time(v)) => builder.append_value(v),
            #[cfg(feature = "dtype-time")]
            (Time(builder), AnyValue::Null) => builder.append_null(),
            (Float32(builder), AnyValue::Null) => builder.append_null(),
            (Float64(builder), AnyValue::Null) => builder.append_null(),
            (Float32(builder), val) => builder.append_value(val.extract()?),
            (Float64(builder), val) => builder.append_value(val.extract()?),
            (Utf8(builder), AnyValue::Utf8(v)) => builder.append_value(v),
            (Utf8(builder), AnyValue::Utf8Owned(v)) => builder.append_value(v),
            (Utf8(builder), AnyValue::Null) => builder.append_null(),
            // Struct and List can be recursive so use anyvalues for that
            (All(_, vals), v) => vals.push(v),

            // dynamic types
            (Utf8(builder), av) => match av {
                AnyValue::Int64(v) => builder.append_value(&format!("{v}")),
                AnyValue::Float64(v) => builder.append_value(&format!("{v}")),
                AnyValue::Boolean(true) => builder.append_value("true"),
                AnyValue::Boolean(false) => builder.append_value("false"),
                _ => return None,
            },
            _ => return None,
        };
        Some(())
    }

    pub(crate) fn add_fallible(&mut self, val: &AnyValue<'a>) -> PolarsResult<()> {
        self.add(val.clone()).ok_or_else(|| {
            PolarsError::ComputeError(format!("Could not append {val:?} to builder; make sure that all rows have the same schema.\n\
            Or consider increasing the the 'schema_inference_length' argument.").into())
        })
    }

    pub fn into_series(self) -> Series {
        use AnyValueBuffer::*;
        match self {
            Boolean(b) => b.finish().into_series(),
            Int32(b) => b.finish().into_series(),
            Int64(b) => b.finish().into_series(),
            UInt32(b) => b.finish().into_series(),
            UInt64(b) => b.finish().into_series(),
            #[cfg(feature = "dtype-date")]
            Date(b) => b.finish().into_date().into_series(),
            #[cfg(feature = "dtype-datetime")]
            Datetime(b, tu, tz) => b.finish().into_datetime(tu, tz).into_series(),
            #[cfg(feature = "dtype-duration")]
            Duration(b, tu) => b.finish().into_duration(tu).into_series(),
            #[cfg(feature = "dtype-time")]
            Time(b) => b.finish().into_time().into_series(),
            Float32(b) => b.finish().into_series(),
            Float64(b) => b.finish().into_series(),
            Utf8(b) => b.finish().into_series(),
            All(dtype, vals) => Series::from_any_values_and_dtype("", &vals, &dtype).unwrap(),
        }
    }

    pub fn new(dtype: &DataType, capacity: usize) -> AnyValueBuffer<'a> {
        (dtype, capacity).into()
    }
}

// datatype and length
impl From<(&DataType, usize)> for AnyValueBuffer<'_> {
    fn from(a: (&DataType, usize)) -> Self {
        let (dt, len) = a;
        use DataType::*;
        match dt {
            Boolean => AnyValueBuffer::Boolean(BooleanChunkedBuilder::new("", len)),
            Int32 => AnyValueBuffer::Int32(PrimitiveChunkedBuilder::new("", len)),
            Int64 => AnyValueBuffer::Int64(PrimitiveChunkedBuilder::new("", len)),
            UInt32 => AnyValueBuffer::UInt32(PrimitiveChunkedBuilder::new("", len)),
            UInt64 => AnyValueBuffer::UInt64(PrimitiveChunkedBuilder::new("", len)),
            #[cfg(feature = "dtype-date")]
            Date => AnyValueBuffer::Date(PrimitiveChunkedBuilder::new("", len)),
            #[cfg(feature = "dtype-datetime")]
            Datetime(tu, tz) => {
                AnyValueBuffer::Datetime(PrimitiveChunkedBuilder::new("", len), *tu, tz.clone())
            }
            #[cfg(feature = "dtype-duration")]
            Duration(tu) => AnyValueBuffer::Duration(PrimitiveChunkedBuilder::new("", len), *tu),
            #[cfg(feature = "dtype-time")]
            Time => AnyValueBuffer::Time(PrimitiveChunkedBuilder::new("", len)),
            Float32 => AnyValueBuffer::Float32(PrimitiveChunkedBuilder::new("", len)),
            Float64 => AnyValueBuffer::Float64(PrimitiveChunkedBuilder::new("", len)),
            Utf8 => AnyValueBuffer::Utf8(Utf8ChunkedBuilder::new("", len, len * 5)),
            // Struct and List can be recursive so use anyvalues for that
            dt => AnyValueBuffer::All(dt.clone(), Vec::with_capacity(len)),
        }
    }
}

#[inline]
unsafe fn add_value<T: NumericNative>(
    values_buf_ptr: usize,
    col_idx: usize,
    row_idx: usize,
    value: T,
) {
    let column = (*(values_buf_ptr as *mut Vec<Vec<T>>)).get_unchecked_mut(col_idx);
    let el_ptr = column.as_mut_ptr();
    *el_ptr.add(row_idx) = value;
}

fn numeric_transpose<T>(cols: &[Series]) -> PolarsResult<DataFrame>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let new_width = cols[0].len();
    let new_height = cols.len();

    let has_nulls = cols.iter().any(|s| s.null_count() > 0);

    let mut values_buf: Vec<Vec<T::Native>> = (0..new_width)
        .map(|_| Vec::with_capacity(new_height))
        .collect();
    let mut validity_buf: Vec<_> = if has_nulls {
        // we first use bools instead of bits, because we can access these in parallel without aliasing
        (0..new_width).map(|_| vec![true; new_height]).collect()
    } else {
        (0..new_width).map(|_| vec![]).collect()
    };

    // work with *mut pointers because we it is UB write to &refs.
    let values_buf_ptr = &mut values_buf as *mut Vec<Vec<T::Native>> as usize;
    let validity_buf_ptr = &mut validity_buf as *mut Vec<Vec<bool>> as usize;

    POOL.install(|| {
        cols.iter().enumerate().for_each(|(row_idx, s)| {
            let s = s.cast(&T::get_dtype()).unwrap();
            let ca = s.unpack::<T>().unwrap();

            // Safety
            // we access in parallel, but every access is unique, so we don't break aliasing rules
            // we also ensured we allocated enough memory, so we never reallocate and thus
            // the pointers remain valid.
            if has_nulls {
                for (col_idx, opt_v) in ca.into_iter().enumerate() {
                    match opt_v {
                        None => unsafe {
                            let column = (*(validity_buf_ptr as *mut Vec<Vec<bool>>))
                                .get_unchecked_mut(col_idx);
                            let el_ptr = column.as_mut_ptr();
                            *el_ptr.add(row_idx) = false;
                            // we must initialize this memory otherwise downstream code
                            // might access uninitialized memory when the masked out values
                            // are changed.
                            add_value(values_buf_ptr, col_idx, row_idx, T::Native::default());
                        },
                        Some(v) => unsafe {
                            add_value(values_buf_ptr, col_idx, row_idx, v);
                        },
                    }
                }
            } else {
                for (col_idx, v) in ca.into_no_null_iter().enumerate() {
                    unsafe {
                        let column = (*(values_buf_ptr as *mut Vec<Vec<T::Native>>))
                            .get_unchecked_mut(col_idx);
                        let el_ptr = column.as_mut_ptr();
                        *el_ptr.add(row_idx) = v;
                    }
                }
            }
        })
    });

    let series = POOL.install(|| {
        values_buf
            .into_par_iter()
            .zip(validity_buf)
            .enumerate()
            .map(|(i, (mut values, validity))| {
                // Safety:
                // all values are written we can now set len
                unsafe {
                    values.set_len(new_height);
                }

                let validity = if has_nulls {
                    let validity = Bitmap::from_trusted_len_iter(validity.iter().copied());
                    if validity.unset_bits() > 0 {
                        Some(validity)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let arr = PrimitiveArray::<T::Native>::new(
                    T::get_dtype().to_arrow(),
                    values.into(),
                    validity,
                );
                let name = format!("column_{i}");
                unsafe {
                    ChunkedArray::<T>::from_chunks(&name, vec![Box::new(arr) as ArrayRef])
                        .into_series()
                }
            })
            .collect()
    });

    Ok(DataFrame::new_no_checks(series))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_transpose() -> PolarsResult<()> {
        let df = df![
            "a" => [1, 2, 3],
            "b" => [10, 20, 30],
        ]?;

        let out = df.transpose()?;
        let expected = df![
            "column_0" => [1, 10],
            "column_1" => [2, 20],
            "column_2" => [3, 30],

        ]?;
        assert!(out.frame_equal_missing(&expected));

        let df = df![
            "a" => [Some(1), None, Some(3)],
            "b" => [Some(10), Some(20), None],
        ]?;
        let out = df.transpose()?;
        let expected = df![
            "column_0" => [1, 10],
            "column_1" => [None, Some(20)],
            "column_2" => [Some(3), None],

        ]?;
        assert!(out.frame_equal_missing(&expected));

        let df = df![
            "a" => ["a", "b", "c"],
            "b" => [Some(10), Some(20), None],
        ]?;
        let out = df.transpose()?;
        let expected = df![
            "column_0" => ["a", "10"],
            "column_1" => ["b", "20"],
            "column_2" => [Some("c"), None],

        ]?;
        assert!(out.frame_equal_missing(&expected));
        Ok(())
    }
}
