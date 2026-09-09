use crate::prelude::*;

impl Series {
    /// Restore `dtype` from its physical representation with safety and Arrow import checks.
    /// Safe counterpart of [`Series::from_physical_unchecked`].
    ///
    /// - `Map`: validate and canonicalize storage before constructing the Map.
    /// - `Categorical` / `Enum`: every code must name a category.
    /// - `Decimal`: validate precision, scale, and value bounds.
    /// - `Object`, `Unknown`: reject reconstruction.
    /// - Temporal types: no per-value checks, matching Arrow import. Out-of-range `Time`
    ///   values may fail during formatting.
    ///
    /// Errors for unsupported dtypes.
    pub fn try_from_physical(&self, dtype: &DataType) -> PolarsResult<Series> {
        let physical = dtype.to_physical();
        polars_ensure!(
            self.dtype() == &physical,
            InvalidOperation:
            "cannot restore `{dtype}` on a Series of type `{}`: its physical type is `{physical}`",
            self.dtype()
        );
        // SAFETY: the physical dtypes match, recursively.
        unsafe { try_from_physical_rec(self, dtype) }
    }
}

/// # Safety
/// `series.dtype()` must equal `dtype.to_physical()`.
unsafe fn try_from_physical_rec(series: &Series, dtype: &DataType) -> PolarsResult<Series> {
    use DataType as D;

    // No logical types remain, even in nested values.
    if series.dtype() == dtype {
        return Ok(series.clone());
    }

    // SAFETY: every recursive call descends into both the physical Series and `dtype`.
    match dtype {
        #[cfg(feature = "dtype-map")]
        D::Map(_, _) => {
            let storage_dtype = dtype.map_storage_dtype().unwrap();
            let storage = unsafe { try_from_physical_rec(series, &storage_dtype)? };
            Ok(MapChunked::try_from_storage(dtype.clone(), storage)?.into_series())
        },
        D::List(inner) => {
            let ca = series.list().unwrap();
            let values = unsafe { try_from_physical_rec(&ca.get_inner(), inner)? };
            Ok(ca.with_inner_values(&values).into_series())
        },
        #[cfg(feature = "dtype-array")]
        D::Array(inner, _) => {
            let ca = series.array().unwrap();
            let values = unsafe { try_from_physical_rec(&ca.get_inner(), inner)? };
            Ok(ca.with_inner_values(&values).into_series())
        },
        #[cfg(feature = "dtype-struct")]
        D::Struct(fields) => {
            let mut dtypes = fields.iter().map(|field| &field.dtype);
            // `try_apply_fields` keeps the outer validity.
            let ca = series.struct_().unwrap().try_apply_fields(|field| unsafe {
                try_from_physical_rec(field, dtypes.next().unwrap())
            })?;
            Ok(ca.into_series())
        },
        #[cfg(feature = "dtype-extension")]
        D::Extension(typ, storage_dtype) => {
            // Check the `into_extension` precondition before reconstruction.
            polars_ensure!(
                !storage_dtype.is_extension(),
                InvalidOperation: "cannot restore `{dtype}`: extension types cannot be nested directly"
            );
            let storage = unsafe { try_from_physical_rec(series, storage_dtype)? };
            Ok(storage.into_extension(typ.clone()))
        },
        #[cfg(feature = "dtype-categorical")]
        D::Categorical(_, _) | D::Enum(_, _) => Series::from_cats_and_dtype(series, dtype, true),
        #[cfg(feature = "dtype-decimal")]
        D::Decimal(precision, scale) => {
            use polars_compute::decimal::{dec128_fits, dec128_verify_prec_scale};

            // Validate precision before `dec128_fits` uses it as a table index.
            dec128_verify_prec_scale(*precision, *scale)?;
            let ca = series.i128()?;
            let fits = ca.downcast_iter().all(|arr| {
                arr.non_null_values_iter()
                    .all(|value| dec128_fits(value, *precision))
            });
            polars_ensure!(
                fits,
                ComputeError: "decimal value does not fit in precision {precision}"
            );
            unsafe { series.from_physical_unchecked(dtype) }
        },
        #[cfg(feature = "object")]
        D::Object(_) => polars_bail!(
            InvalidOperation:
            "cannot restore `{dtype}` from its physical representation: objects are process-local"
        ),
        D::Unknown(_) => polars_bail!(
            InvalidOperation: "cannot restore an unknown dtype from its physical representation"
        ),
        D::Date | D::Datetime(_, _) | D::Duration(_) | D::Time => unsafe {
            series.from_physical_unchecked(dtype)
        },
        _ => polars_bail!(
            InvalidOperation: "cannot validate the physical representation of `{dtype}`"
        ),
    }
}

#[cfg(test)]
mod test {
    use arrow::array::PrimitiveArray;

    use crate::prelude::*;

    #[cfg(feature = "dtype-extension")]
    #[test]
    fn try_from_physical_rejects_directly_nested_extensions() {
        use crate::datatypes::extension::get_extension_type_or_generic;

        let inner = DataType::Extension(
            get_extension_type_or_generic("inner", &DataType::Int64, None),
            Box::new(DataType::Int64),
        );
        let dtype = DataType::Extension(
            get_extension_type_or_generic("outer", &inner, None),
            Box::new(inner),
        );
        let s = Series::new(PlSmallStr::from_static("e"), &[1i64, 2]);
        assert_eq!(s.dtype(), &dtype.to_physical());
        let err = s.try_from_physical(&dtype).err().unwrap();
        assert!(err.to_string().contains("nested directly"), "{err}");
    }

    /// Run with `--no-default-features --features dtype-date` to catch a `Date` arm
    /// accidentally gated on `dtype-time`.
    #[cfg(feature = "dtype-date")]
    #[test]
    fn from_chunk_and_dtype_builds_dates() {
        let chunk = PrimitiveArray::<i32>::from_vec(vec![0, 1]).boxed();
        let s = Series::from_chunk_and_dtype(PlSmallStr::from_static("d"), chunk, &DataType::Date)
            .unwrap();
        assert_eq!(s.dtype(), &DataType::Date);
        assert_eq!(s.len(), 2);
    }

    #[cfg(feature = "dtype-categorical")]
    #[test]
    fn from_chunk_and_dtype_rejects_out_of_range_enum_codes() {
        use polars_dtype::categorical::FrozenCategories;

        let dtype = DataType::from_frozen_categories(FrozenCategories::new(["a", "b"]).unwrap());
        let physical = dtype.to_physical();
        let codes = |codes: &[u32]| {
            Series::new(PlSmallStr::from_static("e"), codes)
                .cast(&physical)
                .unwrap()
                .chunks()[0]
                .clone()
        };

        let err =
            Series::from_chunk_and_dtype(PlSmallStr::from_static("e"), codes(&[0, 7]), &dtype)
                .unwrap_err();
        assert!(err.to_string().contains("invalid category"), "{err}");

        let s = Series::from_chunk_and_dtype(PlSmallStr::from_static("e"), codes(&[0, 1]), &dtype)
            .unwrap();
        assert_eq!(s.dtype(), &dtype);
        assert_eq!(s.null_count(), 0);
    }

    #[cfg(feature = "object")]
    #[test]
    fn from_chunk_and_dtype_rejects_objects_before_reinterpreting() {
        let chunk = PrimitiveArray::<i64>::from_vec(vec![1]).boxed();
        for dtype in [
            DataType::Object("x"),
            DataType::List(Box::new(DataType::Object("x"))),
        ] {
            let err =
                Series::from_chunk_and_dtype(PlSmallStr::from_static("o"), chunk.clone(), &dtype)
                    .unwrap_err();
            assert!(err.to_string().contains("objects"), "{err}");
        }
    }

    #[cfg(feature = "dtype-decimal")]
    #[test]
    fn from_chunk_and_dtype_validates_decimals() {
        let name = PlSmallStr::from_static("d");
        let empty = PrimitiveArray::<i128>::new_empty(ArrowDataType::Int128).boxed();

        // Validate metadata even for empty arrays.
        let err =
            Series::from_chunk_and_dtype(name.clone(), empty.clone(), &DataType::Decimal(50, 2))
                .unwrap_err();
        assert!(err.to_string().contains("precision"), "{err}");
        let err = Series::from_chunk_and_dtype(name.clone(), empty, &DataType::Decimal(5, 9))
            .unwrap_err();
        assert!(err.to_string().contains("scale"), "{err}");

        let too_big = PrimitiveArray::<i128>::from_vec(vec![100_000]).boxed();
        let err = Series::from_chunk_and_dtype(name.clone(), too_big, &DataType::Decimal(5, 0))
            .unwrap_err();
        assert!(err.to_string().contains("does not fit"), "{err}");

        let fits = PrimitiveArray::<i128>::from_vec(vec![99_999]).boxed();
        let s = Series::from_chunk_and_dtype(name, fits, &DataType::Decimal(5, 0)).unwrap();
        assert_eq!(s.dtype(), &DataType::Decimal(5, 0));
    }
}
