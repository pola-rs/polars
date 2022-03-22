use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use crate::utils::get_supertype;
use crate::POOL;

use arrow::bitmap::Bitmap;
use rayon::prelude::*;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Row<'a>(pub Vec<AnyValue<'a>>);

impl<'a> Row<'a> {
    pub fn new(values: Vec<AnyValue<'a>>) -> Self {
        Row(values)
    }
}

impl DataFrame {
    /// Get a row from a DataFrame. Use of this is discouraged as it will likely be slow.
    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    pub fn get_row(&self, idx: usize) -> Row {
        let values = self.columns.iter().map(|s| s.get(idx)).collect::<Vec<_>>();
        Row(values)
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible to make sure that the row has at least the capacity for the number
    /// of columns in the DataFrame
    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    pub fn get_row_amortized<'a>(&'a self, idx: usize, row: &mut Row<'a>) {
        self.columns
            .iter()
            .zip(&mut row.0)
            .for_each(|(s, any_val)| {
                *any_val = s.get(idx);
            });
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible to make sure that the row has at least the capacity for the number
    /// of columns in the DataFrame
    ///
    /// # Safety
    /// Does not do any bounds checking.
    #[inline]
    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
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
    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    pub fn from_rows_and_schema(rows: &[Row], schema: &Schema) -> Result<Self> {
        Self::from_rows_iter_and_schema(rows.iter(), schema)
    }

    /// Create a new DataFrame from an iterator over rows. This should only be used when you have row wise data,
    /// as this is a lot slower than creating the `Series` in a columnar fashion
    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    pub fn from_rows_iter_and_schema<'a, I>(mut rows: I, schema: &Schema) -> Result<Self>
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

        rows.try_for_each::<_, Result<()>>(|row| {
            for (value, buf) in row.0.iter().zip(&mut buffers) {
                buf.add_falible(value)?
            }
            Ok(())
        })?;
        let v = buffers
            .into_iter()
            .zip(schema.iter_names())
            .map(|(b, name)| {
                let mut s = b.into_series();
                s.rename(name);
                s
            })
            .collect();
        DataFrame::new(v)
    }

    /// Create a new DataFrame from rows. This should only be used when you have row wise data,
    /// as this is a lot slower than creating the `Series` in a columnar fashion
    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    pub fn from_rows(rows: &[Row]) -> Result<Self> {
        let schema = rows_to_schema(rows);
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

    pub(crate) fn transpose_from_dtype(&self, dtype: &DataType) -> Result<DataFrame> {
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

                // this is very expensive. A lot of cache misses here.
                // This is the part that is performance critical.
                self.columns.iter().for_each(|s| {
                    let s = s.cast(dtype).unwrap();
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

    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    /// Transpose a DataFrame. This is a very expensive operation.
    pub fn transpose(&self) -> Result<DataFrame> {
        let height = self.height();
        let width = self.width();
        if height == 0 || width == 0 {
            return Err(PolarsError::NoData("empty dataframe".into()));
        }

        let dtype = self.get_supertype().unwrap()?;
        self.transpose_from_dtype(&dtype)
    }
}

type Tracker = PlHashMap<String, PlHashSet<DataType>>;

pub fn infer_schema(
    iter: impl Iterator<Item = Vec<(String, impl Into<DataType>)>>,
    infer_schema_length: usize,
) -> Schema {
    let mut values: Tracker = Tracker::new();
    let len = iter.size_hint().1.unwrap();

    let max_infer = std::cmp::min(len, infer_schema_length);
    for inner in iter.take(max_infer) {
        for (key, value) in inner {
            add_or_insert(&mut values, &key, value.into());
        }
    }
    Schema::from(resolve_fields(values))
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

fn coerce_data_type<A: Borrow<DataType>>(datatypes: &[A]) -> DataType {
    use DataType::*;

    let are_all_equal = datatypes.windows(2).all(|w| w[0].borrow() == w[1].borrow());

    if are_all_equal {
        return datatypes[0].borrow().clone();
    }
    if datatypes.len() > 2 {
        return Utf8;
    }

    let (lhs, rhs) = (datatypes[0].borrow(), datatypes[1].borrow());
    get_supertype(lhs, rhs).unwrap_or(Utf8)
}

/// Infer schema from rows.
pub fn rows_to_schema(rows: &[Row]) -> Schema {
    // no of rows to use to infer dtype
    let max_infer = std::cmp::min(rows.len(), 50);
    let mut schema: Schema = (&rows[0]).into();
    // the first row that has no nulls will be used to infer the schema.
    // if there is a null, we check the next row and see if we can update the schema

    for row in rows.iter().take(max_infer).skip(1) {
        // for i in 1..max_infer {
        let nulls: Vec<_> = schema
            .iter_dtypes()
            .enumerate()
            .filter_map(|(i, dtype)| {
                if matches!(dtype, DataType::Null) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if nulls.is_empty() {
            break;
        } else {
            for i in nulls {
                let dtype = (&row.0[i]).into();
                schema.coerce_by_index(i, dtype).unwrap();
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
impl<'a> From<&AnyValue<'a>> for DataType {
    fn from(val: &AnyValue<'a>) -> Self {
        use AnyValue::*;
        match val {
            Null => DataType::Null,
            Boolean(_) => DataType::Boolean,
            Utf8(_) => DataType::Utf8,
            Utf8Owned(_) => DataType::Utf8,
            UInt32(_) => DataType::UInt32,
            UInt64(_) => DataType::UInt64,
            Int32(_) => DataType::Int32,
            Int64(_) => DataType::Int64,
            Float32(_) => DataType::Float32,
            Float64(_) => DataType::Float64,
            #[cfg(feature = "dtype-date")]
            Date(_) => DataType::Date,
            #[cfg(feature = "dtype-datetime")]
            Datetime(_, tu, tz) => DataType::Datetime(*tu, (*tz).clone()),
            #[cfg(feature = "dtype-time")]
            Time(_) => DataType::Time,
            List(s) => DataType::List(Box::new(s.dtype().clone())),
            _ => unimplemented!(),
        }
    }
}

impl From<&Row<'_>> for Schema {
    fn from(row: &Row) -> Self {
        let fields = row.0.iter().enumerate().map(|(i, av)| {
            let dtype = av.into();
            Field::new(format!("column_{}", i).as_ref(), dtype)
        });

        Schema::from(fields)
    }
}

pub(crate) enum AnyValueBuffer {
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
    #[cfg(feature = "dtype-time")]
    Time(PrimitiveChunkedBuilder<Int64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    Utf8(Utf8ChunkedBuilder),
    List(Box<dyn ListBuilderTrait>),
}

impl Debug for AnyValueBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use AnyValueBuffer::*;
        match self {
            Boolean(_) => f.write_str("boolean"),
            Int32(_) => f.write_str("i32"),
            Int64(_) => f.write_str("i64"),
            UInt32(_) => f.write_str("u32"),
            UInt64(_) => f.write_str("u64"),
            #[cfg(feature = "dtype-date")]
            Date(_) => f.write_str("Date"),
            #[cfg(feature = "dtype-datetime")]
            Datetime(_, _, _) => f.write_str("datetime"),
            #[cfg(feature = "dtype-time")]
            Time(_) => f.write_str("time"),
            Float32(_) => f.write_str("f32"),
            Float64(_) => f.write_str("f64"),
            Utf8(_) => f.write_str("utf8"),
            List(_) => f.write_str("list"),
        }
    }
}

impl AnyValueBuffer {
    pub(crate) fn add(&mut self, val: AnyValue) -> Option<()> {
        use AnyValueBuffer::*;
        match (self, val) {
            (Boolean(builder), AnyValue::Boolean(v)) => builder.append_value(v),
            (Boolean(builder), AnyValue::Null) => builder.append_null(),
            (Int32(builder), AnyValue::Int32(v)) => builder.append_value(v),
            (Int32(builder), AnyValue::Null) => builder.append_null(),
            (Int64(builder), AnyValue::Int64(v)) => builder.append_value(v),
            (Int64(builder), AnyValue::Null) => builder.append_null(),
            (UInt32(builder), AnyValue::UInt32(v)) => builder.append_value(v),
            (UInt32(builder), AnyValue::Null) => builder.append_null(),
            (UInt64(builder), AnyValue::UInt64(v)) => builder.append_value(v),
            (UInt64(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-date")]
            (Date(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-date")]
            (Date(builder), AnyValue::Date(v)) => builder.append_value(v),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(builder, _, _), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(builder, _, _), AnyValue::Datetime(v, _, _)) => builder.append_value(v),
            #[cfg(feature = "dtype-time")]
            (Time(builder), AnyValue::Time(v)) => builder.append_value(v),
            #[cfg(feature = "dtype-time")]
            (Time(builder), AnyValue::Null) => builder.append_null(),
            (Float32(builder), AnyValue::Float32(v)) => builder.append_value(v),
            (Float32(builder), AnyValue::Null) => builder.append_null(),
            (Float64(builder), AnyValue::Float64(v)) => builder.append_value(v),
            (Float64(builder), AnyValue::Null) => builder.append_null(),
            (Utf8(builder), AnyValue::Utf8(v)) => builder.append_value(v),
            (Utf8(builder), AnyValue::Null) => builder.append_null(),
            (List(builder), AnyValue::List(v)) => builder.append_series(&v),
            (List(builder), AnyValue::Null) => builder.append_null(),
            _ => return None,
        };
        Some(())
    }

    pub(crate) fn add_falible(&mut self, val: &AnyValue) -> Result<()> {
        self.add(val.clone()).ok_or_else(|| {
            PolarsError::ComputeError(format!("Could not append {:?} to builder; make sure that all rows have the same schema.", val).into())
        })
    }

    pub(crate) fn into_series(self) -> Series {
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
            #[cfg(feature = "dtype-time")]
            Time(b) => b.finish().into_time().into_series(),
            Float32(b) => b.finish().into_series(),
            Float64(b) => b.finish().into_series(),
            Utf8(b) => b.finish().into_series(),
            List(mut b) => b.finish().into_series(),
        }
    }
}

// datatype and length
impl From<(&DataType, usize)> for AnyValueBuffer {
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
            #[cfg(feature = "dtype-time")]
            Time => AnyValueBuffer::Time(PrimitiveChunkedBuilder::new("", len)),
            Float32 => AnyValueBuffer::Float32(PrimitiveChunkedBuilder::new("", len)),
            Float64 => AnyValueBuffer::Float64(PrimitiveChunkedBuilder::new("", len)),
            Utf8 => AnyValueBuffer::Utf8(Utf8ChunkedBuilder::new("", len, len * 5)),
            List(inner) => AnyValueBuffer::List(get_list_builder(inner, len * 10, len, "")),
            _ => unimplemented!(),
        }
    }
}

fn numeric_transpose<T>(cols: &[Series]) -> Result<DataFrame>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let new_width = cols[0].len();
    let new_height = cols.len();

    let has_nulls = cols.iter().any(|s| s.null_count() > 0);

    let values_buf: Vec<Vec<T::Native>> = (0..new_width)
        .map(|_| Vec::with_capacity(new_height))
        .collect();
    let validity_buf: Vec<_> = if has_nulls {
        // we first use bools instead of bits, because we can access these in parallel without aliasing
        (0..new_width).map(|_| vec![true; new_height]).collect()
    } else {
        (0..new_width).map(|_| vec![]).collect()
    };

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
                            let column = validity_buf.get_unchecked(col_idx);
                            let el_ptr = column.as_ptr() as *mut bool;
                            *el_ptr.add(row_idx) = false;
                        },
                        Some(v) => unsafe {
                            let column = values_buf.get_unchecked(col_idx);
                            let el_ptr = column.as_ptr() as *mut T::Native;
                            *el_ptr.add(row_idx) = v;
                        },
                    }
                }
            } else {
                for (col_idx, v) in ca.into_no_null_iter().enumerate() {
                    unsafe {
                        let column = values_buf.get(col_idx).unwrap();
                        let el_ptr = column.as_ptr() as *mut T::Native;
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
                    if validity.null_count() > 0 {
                        Some(validity)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let arr = PrimitiveArray::<T::Native>::from_data(
                    T::get_dtype().to_arrow(),
                    values.into(),
                    validity,
                );
                let name = format!("column_{}", i);
                ChunkedArray::<T>::from_chunks(&name, vec![Arc::new(arr) as ArrayRef]).into_series()
            })
            .collect()
    });

    Ok(DataFrame::new_no_checks(series))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_transpose() -> Result<()> {
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
