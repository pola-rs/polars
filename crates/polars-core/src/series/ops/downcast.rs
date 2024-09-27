use crate::prelude::*;
use crate::series::implementations::null::NullChunked;

macro_rules! unpack_chunked_err {
    ($series:expr => $name:expr) => {
        polars_err!(SchemaMismatch: "invalid series dtype: expected `{}`, got `{}`", $name, $series.dtype())
    };
}

macro_rules! try_unpack_chunked {
    ($series:expr, $expected:pat => $ca:ty) => {
        match $series.dtype() {
            $expected => {
                // Check downcast in debug compiles
                #[cfg(debug_assertions)]
                {
                    Some($series.as_ref().as_any().downcast_ref::<$ca>().unwrap())
                }
                #[cfg(not(debug_assertions))]
                unsafe {
                    Some(&*($series.as_ref() as *const dyn SeriesTrait as *const $ca))
                }
            },
            _ => None,
        }
    };
}

impl Series {
    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int8]`
    pub fn try_i8(&self) -> Option<&Int8Chunked> {
        try_unpack_chunked!(self, DataType::Int8 => Int8Chunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int16]`
    pub fn try_i16(&self) -> Option<&Int16Chunked> {
        try_unpack_chunked!(self, DataType::Int16 => Int16Chunked)
    }

    /// Unpack to [`ChunkedArray`]
    /// ```
    /// # use polars_core::prelude::*;
    /// let s = Series::new("foo".into(), [1i32 ,2, 3]);
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
    pub fn try_i32(&self) -> Option<&Int32Chunked> {
        try_unpack_chunked!(self, DataType::Int32 => Int32Chunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int64]`
    pub fn try_i64(&self) -> Option<&Int64Chunked> {
        try_unpack_chunked!(self, DataType::Int64 => Int64Chunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Float32]`
    pub fn try_f32(&self) -> Option<&Float32Chunked> {
        try_unpack_chunked!(self, DataType::Float32 => Float32Chunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Float64]`
    pub fn try_f64(&self) -> Option<&Float64Chunked> {
        try_unpack_chunked!(self, DataType::Float64 => Float64Chunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt8]`
    pub fn try_u8(&self) -> Option<&UInt8Chunked> {
        try_unpack_chunked!(self, DataType::UInt8 => UInt8Chunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt16]`
    pub fn try_u16(&self) -> Option<&UInt16Chunked> {
        try_unpack_chunked!(self, DataType::UInt16 => UInt16Chunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt32]`
    pub fn try_u32(&self) -> Option<&UInt32Chunked> {
        try_unpack_chunked!(self, DataType::UInt32 => UInt32Chunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt64]`
    pub fn try_u64(&self) -> Option<&UInt64Chunked> {
        try_unpack_chunked!(self, DataType::UInt64 => UInt64Chunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Boolean]`
    pub fn try_bool(&self) -> Option<&BooleanChunked> {
        try_unpack_chunked!(self, DataType::Boolean => BooleanChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::String]`
    pub fn try_str(&self) -> Option<&StringChunked> {
        try_unpack_chunked!(self, DataType::String => StringChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Binary]`
    pub fn try_binary(&self) -> Option<&BinaryChunked> {
        try_unpack_chunked!(self, DataType::Binary => BinaryChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Binary]`
    pub fn try_binary_offset(&self) -> Option<&BinaryOffsetChunked> {
        try_unpack_chunked!(self, DataType::BinaryOffset => BinaryOffsetChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Time]`
    #[cfg(feature = "dtype-time")]
    pub fn try_time(&self) -> Option<&TimeChunked> {
        try_unpack_chunked!(self, DataType::Time => TimeChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Date]`
    #[cfg(feature = "dtype-date")]
    pub fn try_date(&self) -> Option<&DateChunked> {
        try_unpack_chunked!(self, DataType::Date => DateChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Datetime]`
    #[cfg(feature = "dtype-datetime")]
    pub fn try_datetime(&self) -> Option<&DatetimeChunked> {
        try_unpack_chunked!(self, DataType::Datetime(_, _) => DatetimeChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Duration]`
    #[cfg(feature = "dtype-duration")]
    pub fn try_duration(&self) -> Option<&DurationChunked> {
        try_unpack_chunked!(self, DataType::Duration(_) => DurationChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Decimal]`
    #[cfg(feature = "dtype-decimal")]
    pub fn try_decimal(&self) -> Option<&DecimalChunked> {
        try_unpack_chunked!(self, DataType::Decimal(_, _) => DecimalChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype list
    pub fn try_list(&self) -> Option<&ListChunked> {
        try_unpack_chunked!(self, DataType::List(_) => ListChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Array]`
    #[cfg(feature = "dtype-array")]
    pub fn try_array(&self) -> Option<&ArrayChunked> {
        try_unpack_chunked!(self, DataType::Array(_, _) => ArrayChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Categorical]`
    #[cfg(feature = "dtype-categorical")]
    pub fn try_categorical(&self) -> Option<&CategoricalChunked> {
        try_unpack_chunked!(self, DataType::Categorical(_, _) | DataType::Enum(_, _) => CategoricalChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Struct]`
    #[cfg(feature = "dtype-struct")]
    pub fn try_struct(&self) -> Option<&StructChunked> {
        #[cfg(debug_assertions)]
        {
            if let DataType::Struct(_) = self.dtype() {
                let any = self.as_any();
                assert!(any.is::<StructChunked>());
            }
        }
        try_unpack_chunked!(self, DataType::Struct(_) => StructChunked)
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Null]`
    pub fn try_null(&self) -> Option<&NullChunked> {
        try_unpack_chunked!(self, DataType::Null => NullChunked)
    }
    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int8]`
    pub fn i8(&self) -> PolarsResult<&Int8Chunked> {
        self.try_i8()
            .ok_or_else(|| unpack_chunked_err!(self => "Int8"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int16]`
    pub fn i16(&self) -> PolarsResult<&Int16Chunked> {
        self.try_i16()
            .ok_or_else(|| unpack_chunked_err!(self => "Int16"))
    }

    /// Unpack to [`ChunkedArray`]
    /// ```
    /// # use polars_core::prelude::*;
    /// let s = Series::new("foo".into(), [1i32 ,2, 3]);
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
        self.try_i32()
            .ok_or_else(|| unpack_chunked_err!(self => "Int32"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Int64]`
    pub fn i64(&self) -> PolarsResult<&Int64Chunked> {
        self.try_i64()
            .ok_or_else(|| unpack_chunked_err!(self => "Int64"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Float32]`
    pub fn f32(&self) -> PolarsResult<&Float32Chunked> {
        self.try_f32()
            .ok_or_else(|| unpack_chunked_err!(self => "Float32"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Float64]`
    pub fn f64(&self) -> PolarsResult<&Float64Chunked> {
        self.try_f64()
            .ok_or_else(|| unpack_chunked_err!(self => "Float64"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt8]`
    pub fn u8(&self) -> PolarsResult<&UInt8Chunked> {
        self.try_u8()
            .ok_or_else(|| unpack_chunked_err!(self => "UInt8"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt16]`
    pub fn u16(&self) -> PolarsResult<&UInt16Chunked> {
        self.try_u16()
            .ok_or_else(|| unpack_chunked_err!(self => "UInt16"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt32]`
    pub fn u32(&self) -> PolarsResult<&UInt32Chunked> {
        self.try_u32()
            .ok_or_else(|| unpack_chunked_err!(self => "UInt32"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::UInt64]`
    pub fn u64(&self) -> PolarsResult<&UInt64Chunked> {
        self.try_u64()
            .ok_or_else(|| unpack_chunked_err!(self => "UInt64"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Boolean]`
    pub fn bool(&self) -> PolarsResult<&BooleanChunked> {
        self.try_bool()
            .ok_or_else(|| unpack_chunked_err!(self => "Boolean"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::String]`
    pub fn str(&self) -> PolarsResult<&StringChunked> {
        self.try_str()
            .ok_or_else(|| unpack_chunked_err!(self => "String"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Binary]`
    pub fn binary(&self) -> PolarsResult<&BinaryChunked> {
        self.try_binary()
            .ok_or_else(|| unpack_chunked_err!(self => "Binary"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Binary]`
    pub fn binary_offset(&self) -> PolarsResult<&BinaryOffsetChunked> {
        self.try_binary_offset()
            .ok_or_else(|| unpack_chunked_err!(self => "BinaryOffset"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Time]`
    #[cfg(feature = "dtype-time")]
    pub fn time(&self) -> PolarsResult<&TimeChunked> {
        self.try_time()
            .ok_or_else(|| unpack_chunked_err!(self => "Time"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Date]`
    #[cfg(feature = "dtype-date")]
    pub fn date(&self) -> PolarsResult<&DateChunked> {
        self.try_date()
            .ok_or_else(|| unpack_chunked_err!(self => "Date"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Datetime]`
    #[cfg(feature = "dtype-datetime")]
    pub fn datetime(&self) -> PolarsResult<&DatetimeChunked> {
        self.try_datetime()
            .ok_or_else(|| unpack_chunked_err!(self => "Datetime"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Duration]`
    #[cfg(feature = "dtype-duration")]
    pub fn duration(&self) -> PolarsResult<&DurationChunked> {
        self.try_duration()
            .ok_or_else(|| unpack_chunked_err!(self => "Duration"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Decimal]`
    #[cfg(feature = "dtype-decimal")]
    pub fn decimal(&self) -> PolarsResult<&DecimalChunked> {
        self.try_decimal()
            .ok_or_else(|| unpack_chunked_err!(self => "Decimal"))
    }

    /// Unpack to [`ChunkedArray`] of dtype list
    pub fn list(&self) -> PolarsResult<&ListChunked> {
        self.try_list()
            .ok_or_else(|| unpack_chunked_err!(self => "List"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Array]`
    #[cfg(feature = "dtype-array")]
    pub fn array(&self) -> PolarsResult<&ArrayChunked> {
        self.try_array()
            .ok_or_else(|| unpack_chunked_err!(self => "FixedSizeList"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Categorical]`
    #[cfg(feature = "dtype-categorical")]
    pub fn categorical(&self) -> PolarsResult<&CategoricalChunked> {
        self.try_categorical()
            .ok_or_else(|| unpack_chunked_err!(self => "Enum | Categorical"))
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

        self.try_struct()
            .ok_or_else(|| unpack_chunked_err!(self => "Struct"))
    }

    /// Unpack to [`ChunkedArray`] of dtype `[DataType::Null]`
    pub fn null(&self) -> PolarsResult<&NullChunked> {
        self.try_null()
            .ok_or_else(|| unpack_chunked_err!(self => "Null"))
    }
}
