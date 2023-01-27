use crate::prelude::*;

impl Series {
    /// Unpack to ChunkedArray of dtype i8
    pub fn i8(&self) -> PolarsResult<&Int8Chunked> {
        match self.dtype() {
            DataType::Int8 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const Int8Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Int8").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray i16
    pub fn i16(&self) -> PolarsResult<&Int16Chunked> {
        match self.dtype() {
            DataType::Int16 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const Int16Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Int16").into(),
            )),
        }
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
        match self.dtype() {
            DataType::Int32 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const Int32Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Int32").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype i64
    pub fn i64(&self) -> PolarsResult<&Int64Chunked> {
        match self.dtype() {
            DataType::Int64 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const Int64Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Int64").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype f32
    pub fn f32(&self) -> PolarsResult<&Float32Chunked> {
        match self.dtype() {
            DataType::Float32 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const Float32Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Float32").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype f64
    pub fn f64(&self) -> PolarsResult<&Float64Chunked> {
        match self.dtype() {
            DataType::Float64 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const Float64Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Float64").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype u8
    pub fn u8(&self) -> PolarsResult<&UInt8Chunked> {
        match self.dtype() {
            DataType::UInt8 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const UInt8Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != UInt8").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype u16
    pub fn u16(&self) -> PolarsResult<&UInt16Chunked> {
        match self.dtype() {
            DataType::UInt16 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const UInt16Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != UInt16").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype u32
    pub fn u32(&self) -> PolarsResult<&UInt32Chunked> {
        match self.dtype() {
            DataType::UInt32 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const UInt32Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != UInt32").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype u64
    pub fn u64(&self) -> PolarsResult<&UInt64Chunked> {
        match self.dtype() {
            DataType::UInt64 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const UInt64Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != UInt64").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype bool
    pub fn bool(&self) -> PolarsResult<&BooleanChunked> {
        match self.dtype() {
            DataType::Boolean => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const BooleanChunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Boolean").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype utf8
    pub fn utf8(&self) -> PolarsResult<&Utf8Chunked> {
        match self.dtype() {
            DataType::Utf8 => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const Utf8Chunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Utf8").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype binary
    #[cfg(feature = "dtype-binary")]
    pub fn binary(&self) -> PolarsResult<&BinaryChunked> {
        match self.dtype() {
            DataType::Binary => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const BinaryChunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != binary").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype Time
    #[cfg(feature = "dtype-time")]
    pub fn time(&self) -> PolarsResult<&TimeChunked> {
        match self.dtype() {
            DataType::Time => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const TimeChunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Time").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype Date
    #[cfg(feature = "dtype-date")]
    pub fn date(&self) -> PolarsResult<&DateChunked> {
        match self.dtype() {
            DataType::Date => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const DateChunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Date").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype datetime
    #[cfg(feature = "dtype-datetime")]
    pub fn datetime(&self) -> PolarsResult<&DatetimeChunked> {
        match self.dtype() {
            DataType::Datetime(_, _) => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const DatetimeChunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Datetime").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype duration
    #[cfg(feature = "dtype-duration")]
    pub fn duration(&self) -> PolarsResult<&DurationChunked> {
        match self.dtype() {
            DataType::Duration(_) => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const DurationChunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Duration").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype list
    pub fn list(&self) -> PolarsResult<&ListChunked> {
        match self.dtype() {
            DataType::List(_) => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const ListChunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != List").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype categorical
    #[cfg(feature = "dtype-categorical")]
    pub fn categorical(&self) -> PolarsResult<&CategoricalChunked> {
        match self.dtype() {
            DataType::Categorical(_) => unsafe {
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const CategoricalChunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Categorical").into(),
            )),
        }
    }

    /// Unpack to ChunkedArray of dtype struct
    #[cfg(feature = "dtype-struct")]
    pub fn struct_(&self) -> PolarsResult<&StructChunked> {
        match self.dtype() {
            DataType::Struct(_) => unsafe {
                #[cfg(debug_assertions)]
                {
                    let any = self.as_any();
                    assert!(any.is::<StructChunked>());
                }
                // Safety
                // We just checked type
                Ok(&*(self.as_ref() as *const dyn SeriesTrait as *const StructChunked))
            },
            dt => Err(PolarsError::SchemaMisMatch(
                format!("Series of dtype: {dt:?} != Struct").into(),
            )),
        }
    }
}
