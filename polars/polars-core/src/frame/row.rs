use crate::prelude::*;
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
        let capacity = rows.len();
        let mut buffers: Vec<_> = schema
            .fields()
            .iter()
            .map(|fld| {
                let buf: Buffer = (fld.data_type(), capacity).into();
                buf
            })
            .collect();

        rows.iter().try_for_each::<_, Result<()>>(|row| {
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
            Date32(_) => Field::new("", DataType::Date32),
            Date64(_) => Field::new("", DataType::Date64),
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
    #[cfg(feature = "dtype-u64")]
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    #[cfg(feature = "dtype-date32")]
    Date32(PrimitiveChunkedBuilder<Date32Type>),
    #[cfg(feature = "dtype-date64")]
    Date64(PrimitiveChunkedBuilder<Date64Type>),
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
            #[cfg(feature = "dtype-u64")]
            UInt64(_) => f.write_str("u64"),
            #[cfg(feature = "dtype-date32")]
            Date32(_) => f.write_str("date32"),
            #[cfg(feature = "dtype-date64")]
            Date64(_) => f.write_str("date64"),
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
            #[cfg(feature = "dtype-u64")]
            (UInt64(builder), AnyValue::UInt64(v)) => builder.append_value(v),
            #[cfg(feature = "dtype-u64")]
            (UInt64(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-date32")]
            (Date32(builder), AnyValue::Null) => builder.append_null(),
            #[cfg(feature = "dtype-date64")]
            (Date64(builder), AnyValue::Date64(v)) => builder.append_value(v),
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
            #[cfg(feature = "dtype-u64")]
            UInt64(b) => b.finish().into_series(),
            #[cfg(feature = "dtype-date32")]
            Date32(b) => b.finish().into_series(),
            #[cfg(feature = "dtype-date64")]
            Date64(b) => b.finish().into_series(),
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
            #[cfg(feature = "dtype-u64")]
            UInt64 => Buffer::UInt64(PrimitiveChunkedBuilder::new("", len)),
            #[cfg(feature = "dtype-date32")]
            Date32 => Buffer::Date32(PrimitiveChunkedBuilder::new("", len)),
            #[cfg(feature = "dtype-date64")]
            Date64 => Buffer::Date64(PrimitiveChunkedBuilder::new("", len)),
            Float32 => Buffer::Float32(PrimitiveChunkedBuilder::new("", len)),
            Float64 => Buffer::Float64(PrimitiveChunkedBuilder::new("", len)),
            Utf8 => Buffer::Utf8(Utf8ChunkedBuilder::new("", len, len * 5)),
            _ => unimplemented!(),
        }
    }
}
