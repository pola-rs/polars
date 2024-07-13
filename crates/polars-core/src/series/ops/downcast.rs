use crate::prelude::*;
use crate::series::implementations::null::NullChunked;

macro_rules! unpack_chunked {
    ($series:expr, $expected:pat => $ca:ty, $name:expr) => {
        match $series.dtype() {
            $expected => {
                // Check downcast in debug compiles
                #[cfg(debug_assertions)]
                {
                    Ok($series.as_ref().as_any().downcast_ref::<$ca>().unwrap())
                }
                #[cfg(not(debug_assertions))]
                unsafe {
                    Ok(&*($series.as_ref() as *const dyn SeriesTrait as *const $ca))
                }
            },
            dt => polars_bail!(
                SchemaMismatch: "invalid series dtype: expected `{}`, got `{}`", $name, dt,
            ),
        }
    };
}

impl Series {
    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int8]`
    pub fn i8(&self) -> PolarsResult<&Int8Chunked> {
        unpack_chunked!(self, DataType::Int8 => Int8Chunked, "Int8")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int16]`
    pub fn i16(&self) -> PolarsResult<&Int16Chunked> {
        unpack_chunked!(self, DataType::Int16 => Int16Chunked, "Int16")
    }

    /// Unpack to [`ChunkedArray`]
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
    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int32]`
    pub fn i32(&self) -> PolarsResult<&Int32Chunked> {
        unpack_chunked!(self, DataType::Int32 => Int32Chunked, "Int32")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int64]`
    pub fn i64(&self) -> PolarsResult<&Int64Chunked> {
        unpack_chunked!(self, DataType::Int64 => Int64Chunked, "Int64")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Float32]`
    pub fn f32(&self) -> PolarsResult<&Float32Chunked> {
        unpack_chunked!(self, DataType::Float32 => Float32Chunked, "Float32")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Float64]`
    pub fn f64(&self) -> PolarsResult<&Float64Chunked> {
        unpack_chunked!(self, DataType::Float64 => Float64Chunked, "Float64")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt8]`
    pub fn u8(&self) -> PolarsResult<&UInt8Chunked> {
        unpack_chunked!(self, DataType::UInt8 => UInt8Chunked, "UInt8")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt16]`
    pub fn u16(&self) -> PolarsResult<&UInt16Chunked> {
        unpack_chunked!(self, DataType::UInt16 => UInt16Chunked, "UInt16")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt32]`
    pub fn u32(&self) -> PolarsResult<&UInt32Chunked> {
        unpack_chunked!(self, DataType::UInt32 => UInt32Chunked, "UInt32")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt64]`
    pub fn u64(&self) -> PolarsResult<&UInt64Chunked> {
        unpack_chunked!(self, DataType::UInt64 => UInt64Chunked, "UInt64")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Boolean]`
    pub fn bool(&self) -> PolarsResult<&BooleanChunked> {
        unpack_chunked!(self, DataType::Boolean => BooleanChunked, "Boolean")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::String]`
    pub fn str(&self) -> PolarsResult<&StringChunked> {
        unpack_chunked!(self, DataType::String => StringChunked, "String")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Binary]`
    pub fn binary(&self) -> PolarsResult<&BinaryChunked> {
        unpack_chunked!(self, DataType::Binary => BinaryChunked, "Binary")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Binary]`
    pub fn binary_offset(&self) -> PolarsResult<&BinaryOffsetChunked> {
        unpack_chunked!(self, DataType::BinaryOffset => BinaryOffsetChunked, "BinaryOffset")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Time]`
    #[cfg(feature = "dtype-time")]
    pub fn time(&self) -> PolarsResult<&TimeChunked> {
        unpack_chunked!(self, DataType::Time => TimeChunked, "Time")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Date]`
    #[cfg(feature = "dtype-date")]
    pub fn date(&self) -> PolarsResult<&DateChunked> {
        unpack_chunked!(self, DataType::Date => DateChunked, "Date")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Datetime]`
    #[cfg(feature = "dtype-datetime")]
    pub fn datetime(&self) -> PolarsResult<&DatetimeChunked> {
        unpack_chunked!(self, DataType::Datetime(_, _) => DatetimeChunked, "Datetime")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Duration]`
    #[cfg(feature = "dtype-duration")]
    pub fn duration(&self) -> PolarsResult<&DurationChunked> {
        unpack_chunked!(self, DataType::Duration(_) => DurationChunked, "Duration")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Decimal]`
    #[cfg(feature = "dtype-decimal")]
    pub fn decimal(&self) -> PolarsResult<&DecimalChunked> {
        unpack_chunked!(self, DataType::Decimal(_, _) => DecimalChunked, "Decimal")
    }

    /// Unpack to [`ChunkedArray`] of dtype list
    pub fn list(&self) -> PolarsResult<&ListChunked> {
        unpack_chunked!(self, DataType::List(_) => ListChunked, "List")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Array]`
    #[cfg(feature = "dtype-array")]
    pub fn array(&self) -> PolarsResult<&ArrayChunked> {
        unpack_chunked!(self, DataType::Array(_, _) => ArrayChunked, "FixedSizeList")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Categorical]`
    #[cfg(feature = "dtype-categorical")]
    pub fn categorical(&self) -> PolarsResult<&CategoricalChunked> {
        unpack_chunked!(self, DataType::Categorical(_, _) | DataType::Enum(_, _) => CategoricalChunked, "Enum | Categorical")
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Struct]`
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

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Null]`
    pub fn null(&self) -> PolarsResult<&NullChunked> {
        unpack_chunked!(self, DataType::Null => NullChunked, "Null")
    }
}
