use crate::prelude::*;

macro_rules! unpack_chunked {
    ($series:expr, $expected:pat => $ca:ty, $name:expr) => {
        match $series.dtype() {
            $expected => unsafe {
                Ok(&*($series.as_ref() as *const dyn SeriesTrait as *const $ca))
            },
            dt => polars_bail!(
                SchemaMismatch: "invalid series dtype: expected `{}`, got `{}`", $name, dt,
            ),
        }
    };
}

impl Series {
    /// Unpack to ChunkedArray of dtype i8
    pub fn i8(&self) -> PolarsResult<&Int8Chunked> {
        unpack_chunked!(self, DataType::Int8 => Int8Chunked, "Int8")
    }

    /// Unpack to ChunkedArray i16
    pub fn i16(&self) -> PolarsResult<&Int16Chunked> {
        unpack_chunked!(self, DataType::Int16 => Int16Chunked, "Int16")
    }

    /// Unpack to ChunkedArray
    /// ```
    /// # use polars_core::prelude::*;
    /// let s = Series::new("foo", [1i32 ,2, 3]);
    /// let s_squared: Series = s.i32()
    ///     .unwrap()
    ///     .into_iter()
    ///     .map(|opt_v| {
    ///         match opt_v {
    ///             Some(v) => Some(v * v),
    ///             None => None, // null value
    ///         }
    /// }).collect();
    /// ```
    pub fn i32(&self) -> PolarsResult<&Int32Chunked> {
        unpack_chunked!(self, DataType::Int32 => Int32Chunked, "Int32")
    }

    /// Unpack to ChunkedArray of dtype i64
    pub fn i64(&self) -> PolarsResult<&Int64Chunked> {
        unpack_chunked!(self, DataType::Int64 => Int64Chunked, "Int64")
    }

    /// Unpack to ChunkedArray of dtype f32
    pub fn f32(&self) -> PolarsResult<&Float32Chunked> {
        unpack_chunked!(self, DataType::Float32 => Float32Chunked, "Float32")
    }

    /// Unpack to ChunkedArray of dtype f64
    pub fn f64(&self) -> PolarsResult<&Float64Chunked> {
        unpack_chunked!(self, DataType::Float64 => Float64Chunked, "Float64")
    }

    /// Unpack to ChunkedArray of dtype u8
    pub fn u8(&self) -> PolarsResult<&UInt8Chunked> {
        unpack_chunked!(self, DataType::UInt8 => UInt8Chunked, "UInt8")
    }

    /// Unpack to ChunkedArray of dtype u16
    pub fn u16(&self) -> PolarsResult<&UInt16Chunked> {
        unpack_chunked!(self, DataType::UInt16 => UInt16Chunked, "UInt16")
    }

    /// Unpack to ChunkedArray of dtype u32
    pub fn u32(&self) -> PolarsResult<&UInt32Chunked> {
        unpack_chunked!(self, DataType::UInt32 => UInt32Chunked, "UInt32")
    }

    /// Unpack to ChunkedArray of dtype u64
    pub fn u64(&self) -> PolarsResult<&UInt64Chunked> {
        unpack_chunked!(self, DataType::UInt64 => UInt64Chunked, "UInt64")
    }

    /// Unpack to ChunkedArray of dtype bool
    pub fn bool(&self) -> PolarsResult<&BooleanChunked> {
        unpack_chunked!(self, DataType::Boolean => BooleanChunked, "Boolean")
    }

    /// Unpack to ChunkedArray of dtype utf8
    pub fn utf8(&self) -> PolarsResult<&Utf8Chunked> {
        unpack_chunked!(self, DataType::Utf8 => Utf8Chunked, "Utf8")
    }

    /// Unpack to ChunkedArray of dtype binary
    pub fn binary(&self) -> PolarsResult<&BinaryChunked> {
        unpack_chunked!(self, DataType::Binary => BinaryChunked, "Binary")
    }

    /// Unpack to ChunkedArray of dtype Time
    #[cfg(feature = "dtype-time")]
    pub fn time(&self) -> PolarsResult<&TimeChunked> {
        unpack_chunked!(self, DataType::Time => TimeChunked, "Time")
    }

    /// Unpack to ChunkedArray of dtype Date
    #[cfg(feature = "dtype-date")]
    pub fn date(&self) -> PolarsResult<&DateChunked> {
        unpack_chunked!(self, DataType::Date => DateChunked, "Date")
    }

    /// Unpack to ChunkedArray of dtype datetime
    #[cfg(feature = "dtype-datetime")]
    pub fn datetime(&self) -> PolarsResult<&DatetimeChunked> {
        unpack_chunked!(self, DataType::Datetime(_, _) => DatetimeChunked, "Datetime")
    }

    /// Unpack to ChunkedArray of dtype duration
    #[cfg(feature = "dtype-duration")]
    pub fn duration(&self) -> PolarsResult<&DurationChunked> {
        unpack_chunked!(self, DataType::Duration(_) => DurationChunked, "Duration")
    }

    /// Unpack to ChunkedArray of dtype decimal
    #[cfg(feature = "dtype-decimal")]
    pub fn decimal(&self) -> PolarsResult<&DecimalChunked> {
        unpack_chunked!(self, DataType::Decimal(_, _) => DecimalChunked, "Decimal")
    }

    /// Unpack to ChunkedArray of dtype list
    pub fn list(&self) -> PolarsResult<&ListChunked> {
        unpack_chunked!(self, DataType::List(_) => ListChunked, "List")
    }

    /// Unpack to ChunkedArray of dtype categorical
    #[cfg(feature = "dtype-categorical")]
    pub fn categorical(&self) -> PolarsResult<&CategoricalChunked> {
        unpack_chunked!(self, DataType::Categorical(_) => CategoricalChunked, "Categorical")
    }

    /// Unpack to ChunkedArray of dtype struct
    #[cfg(feature = "dtype-struct")]
    pub fn struct_(&self) -> PolarsResult<&StructChunked> {
        #[cfg(debug_assertions)]
        {
            if let DataType::Struct(_) = self.dtype() {
                let any = self.as_any();
                assert!(any.is::<StructChunked>());
            }
        }
        unpack_chunked!(self, DataType::Struct(_) => StructChunked, "Struct")
    }
}
