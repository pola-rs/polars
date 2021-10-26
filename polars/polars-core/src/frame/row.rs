use crate::prelude::*;
use crate::utils::get_supertype;
use itertools::Itertools;
use std::fmt::{Debug, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub struct Row<'a>(pub Vec<AnyValue<'a>>);

impl DataFrame {
    /// Get a row from a DataFrame. Use of this is discouraged as it will likely be slow.
    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    pub fn get_row(&self, idx: usize) -> Row {
        let values = self.columns.iter().map(|s| s.get(idx)).collect_vec();
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

    fn from_rows_iter_and_schema<'a, I>(mut rows: I, schema: &Schema) -> Result<Self>
    where
        I: Iterator<Item = &'a Row<'a>>,
    {
        let capacity = rows.size_hint().0;

        let mut buffers: Vec<_> = schema
            .fields()
            .iter()
            .map(|fld| {
                let buf: Buffer = (fld.data_type(), capacity).into();
                buf
            })
            .collect();

        rows.try_for_each::<_, Result<()>>(|row| {
            for (value, buf) in row.0.iter().zip(&mut buffers) {
                buf.add(value.clone())?
            }
            Ok(())
        })?;
        let v = buffers
            .into_iter()
            .zip(schema.fields())
            .map(|(b, fld)| {
                let mut s = b.into_series();
                s.rename(fld.name());
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
            .fields()
            .iter()
            .any(|fld| matches!(fld.data_type(), DataType::Null));
        if has_nulls {
            return Err(PolarsError::HasNullValues(
                "Could not infer row types, because of the null values".into(),
            ));
        }
        Self::from_rows_and_schema(rows, &schema)
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "rows")))]
    /// Transpose a DataFrame. This is a very expensive operation.
    pub fn transpose(&self) -> Result<DataFrame> {
        // TODO: if needed we could optimize this to specialized builders. Now we use AnyValue even though
        // we cast all Series to same dtype. So we could patter match en create at once.

        let height = self.height();
        if height == 0 || self.width() == 0 {
            return Err(PolarsError::NoData("empty dataframe".into()));
        }

        let dtype = self
            .columns
            .iter()
            .map(|s| Ok(s.dtype().clone()))
            .reduce(|a, b| get_supertype(&a?, &b?))
            .unwrap()?;

        let schema = Schema::new(
            (0..height)
                .map(|i| Field::new(format!("column_{}", i).as_ref(), dtype.clone()))
                .collect(),
        );

        let row_container = vec![AnyValue::Null; height];
        let mut row = Row(row_container);
        let row_ptr = &row as *const Row;

        let columns = self
            .columns
            .iter()
            .map(|s| s.cast(&dtype))
            .collect::<Result<Vec<_>>>()?;

        let iter = columns.iter().map(|s| {
            (0..s.len()).zip(row.0.iter_mut()).for_each(|(i, av)| {
                // Safety:
                // we iterate over the length of s, so we are in bounds
                unsafe { *av = s.get_unchecked(i) };
            });
            // borrow checker does not allow row borrow, so we deref from raw ptr.
            // we do all this to amortize allocs
            // Safety:
            // row is still alive
            unsafe { &*row_ptr }
        });
        Self::from_rows_iter_and_schema(iter, &schema)
    }
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
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                if matches!(f.data_type(), DataType::Null) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if nulls.is_empty() {
            break;
        } else {
            let fields = schema.fields_mut();
            let local_schema: Schema = row.into();
            for i in nulls {
                fields[i] = local_schema.fields()[i].clone()
            }
        }
    }
    schema
}

impl<'a> From<&AnyValue<'a>> for Field {
    fn from(val: &AnyValue<'a>) -> Self {
        use AnyValue::*;
        match val {
            Null => Field::new("", DataType::Null),
            Boolean(_) => Field::new("", DataType::Boolean),
            Utf8(_) => Field::new("", DataType::Utf8),
            UInt32(_) => Field::new("", DataType::UInt32),
            UInt64(_) => Field::new("", DataType::UInt64),
            Int32(_) => Field::new("", DataType::Int32),
            Int64(_) => Field::new("", DataType::Int64),
            Float32(_) => Field::new("", DataType::Float32),
            Float64(_) => Field::new("", DataType::Float64),
            #[cfg(feature = "dtype-date")]
            Date(_) => Field::new("", DataType::Date),
            #[cfg(feature = "dtype-datetime")]
            Datetime(_) => Field::new("", DataType::Datetime),
            #[cfg(feature = "dtype-time")]
            Time(_) => Field::new("", DataType::Time),
            _ => unimplemented!(),
        }
    }
}

impl From<&Row<'_>> for Schema {
    fn from(row: &Row) -> Self {
        let fields = row
            .0
            .iter()
            .enumerate()
            .map(|(i, av)| {
                let field: Field = av.into();
                Field::new(format!("column_{}", i).as_ref(), field.data_type().clone())
            })
            .collect();

        Schema::new(fields)
    }
}

pub(crate) enum Buffer {
    Boolean(BooleanChunkedBuilder),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    #[cfg(feature = "dtype-date")]
    Date(PrimitiveChunkedBuilder<Int32Type>),
    #[cfg(feature = "dtype-datetime")]
    Datetime(PrimitiveChunkedBuilder<Int64Type>),
    #[cfg(feature = "dtype-time")]
    Time(PrimitiveChunkedBuilder<Int64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    Utf8(Utf8ChunkedBuilder),
}

impl Debug for Buffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Buffer::*;
        match self {
            Boolean(_) => f.write_str("boolean"),
            Int32(_) => f.write_str("i32"),
            Int64(_) => f.write_str("i64"),
            UInt32(_) => f.write_str("u32"),
            UInt64(_) => f.write_str("u64"),
            #[cfg(feature = "dtype-date")]
            Date(_) => f.write_str("Date"),
            #[cfg(feature = "dtype-datetime")]
            Datetime(_) => f.write_str("datetime"),
            #[cfg(feature = "dtype-time")]
            Time(_) => f.write_str("time"),
            Float32(_) => f.write_str("f32"),
            Float64(_) => f.write_str("f64"),
            Utf8(_) => f.write_str("utf8"),
        }
    }
}

impl Buffer {
    fn add(&mut self, val: AnyValue) -> Result<()> {
        use Buffer::*;
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
            #[cfg(feature = "dtype-datetime")]
            (Datetime(builder), AnyValue::Datetime(v)) => builder.append_value(v),
            #[cfg(feature = "dtype-time")]
            (Time(builder), AnyValue::Time(v)) => builder.append_value(v),
            (Float32(builder), AnyValue::Null) => builder.append_null(),
            (Float64(builder), AnyValue::Float64(v)) => builder.append_value(v),
            (Utf8(builder), AnyValue::Utf8(v)) => builder.append_value(v),
            (Utf8(builder), AnyValue::Null) => builder.append_null(),
            (buf, val) => return Err(PolarsError::ValueError(format!("Could not append {:?} to builder {:?}; make sure that all rows have the same schema.", val, std::mem::discriminant(buf)).into()))
        };

        Ok(())
    }

    fn into_series(self) -> Series {
        use Buffer::*;
        match self {
            Boolean(b) => b.finish().into_series(),
            Int32(b) => b.finish().into_series(),
            Int64(b) => b.finish().into_series(),
            UInt32(b) => b.finish().into_series(),
            UInt64(b) => b.finish().into_series(),
            #[cfg(feature = "dtype-date")]
            Date(b) => b.finish().into_date().into_series(),
            #[cfg(feature = "dtype-datetime")]
            Datetime(b) => b.finish().into_date().into_series(),
            #[cfg(feature = "dtype-time")]
            Time(b) => b.finish().into_date().into_series(),
            Float32(b) => b.finish().into_series(),
            Float64(b) => b.finish().into_series(),
            Utf8(b) => b.finish().into_series(),
        }
    }
}

// datatype and length
impl From<(&DataType, usize)> for Buffer {
    fn from(a: (&DataType, usize)) -> Self {
        let (dt, len) = a;
        use DataType::*;
        match dt {
            Boolean => Buffer::Boolean(BooleanChunkedBuilder::new("", len)),
            Int32 => Buffer::Int32(PrimitiveChunkedBuilder::new("", len)),
            Int64 => Buffer::Int64(PrimitiveChunkedBuilder::new("", len)),
            UInt32 => Buffer::UInt32(PrimitiveChunkedBuilder::new("", len)),
            UInt64 => Buffer::UInt64(PrimitiveChunkedBuilder::new("", len)),
            #[cfg(feature = "dtype-date")]
            Date => Buffer::Date(PrimitiveChunkedBuilder::new("", len)),
            #[cfg(feature = "dtype-datetime")]
            Datetime => Buffer::Datetime(PrimitiveChunkedBuilder::new("", len)),
            #[cfg(feature = "dtype-time")]
            Time => Buffer::Time(PrimitiveChunkedBuilder::new("", len)),
            Float32 => Buffer::Float32(PrimitiveChunkedBuilder::new("", len)),
            Float64 => Buffer::Float64(PrimitiveChunkedBuilder::new("", len)),
            Utf8 => Buffer::Utf8(Utf8ChunkedBuilder::new("", len, len * 5)),
            _ => unimplemented!(),
        }
    }
}
