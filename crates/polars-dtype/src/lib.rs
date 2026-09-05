//! The data type of a Polars column, and the pieces a data type is built out of.
//!
//! This crate holds the [`DataType`] enum itself and what it names — [`Field`], [`TimeUnit`],
//! [`TimeZone`], the categorical mappings and the extension types. It sits *below*
//! `polars-compute`, so a kernel can be dispatched on a Polars type rather than on the Arrow type
//! that type happens to be laid out as.
//!
//! What stays in `polars-core` is the `PolarsDataType` trait family — `PolarsNumericType` and
//! friends — which is bounded on the arithmetic kernels of `polars-compute` and so cannot live
//! below it. That family is about the *static* type of a `ChunkedArray`; this crate is about the
//! runtime one.

#[cfg(any(feature = "serde", feature = "serde-lazy", feature = "dsl-schema"))]
mod _serde;
#[cfg(all(feature = "dtype-categorical", any(feature = "serde", feature = "serde-lazy", feature = "dsl-schema")))]
mod categories_serde;
pub mod categorical;
pub mod dtype;
#[cfg(feature = "dtype-extension")]
pub mod extension;
pub mod field;
pub mod temporal;

pub use dtype::{CompatLevel, DataType, MetaDataExt, UnknownKind};
pub use field::Field;
pub use temporal::time_unit::TimeUnit;
pub use temporal::time_zone::TimeZone;

/// Reading the environment the way `polars-core` does, without depending on it.
pub(crate) mod config {
    use polars_error::{PolarsResult, polars_ensure};

    #[cfg(feature = "timezones")]
    pub fn verbose() -> bool {
        polars_config::config().verbose()
    }

    /// The interval types of Arrow have no Polars type of their own, so importing one as the
    /// struct that holds its parts is opt-in.
    pub fn check_allow_importing_interval_as_struct(type_name: &'static str) -> PolarsResult<()> {
        polars_ensure!(
            polars_config::config().import_interval_as_struct(),
            ComputeError:
            "could not import from `{type_name}` type. \
            Hint: This can be imported by setting \
            POLARS_IMPORT_INTERVAL_AS_STRUCT=1 in the environment. \
            Note however that this is unstable functionality \
            that may change at any time."
        );
        Ok(())
    }
}

/// The [`DataType`] of the smallest integer a dynamic integer literal fits in.
///
/// This is the type of `materialize_dyn_int(v)` in `polars-core`, which builds the `AnyValue` this
/// answers the type of. The two are pinned to each other by a test there: an `AnyValue` is a value
/// and lives with the values, while a [`DataType`] is needed down here.
pub fn dyn_int_dtype(v: i128) -> DataType {
    // Smallest first, matching `materialize_dyn_int`.
    if i32::try_from(v).is_ok() {
        return DataType::Int32;
    }
    if i64::try_from(v).is_ok() {
        return DataType::Int64;
    }
    if u64::try_from(v).is_ok() {
        return DataType::UInt64;
    }
    #[cfg(feature = "dtype-i128")]
    {
        DataType::Int128
    }
    #[cfg(not(feature = "dtype-i128"))]
    {
        DataType::Null
    }
}

/// The Arrow type the values of an [`Object`](DataType::Object) column are laid out as.
///
/// An object is whatever the host language registered, so the type is only known once that has
/// happened. `polars-core` registers it alongside the rest of the object registry, which holds the
/// builders and converters this crate has no business knowing about — only the type reaches here.
#[cfg(feature = "object")]
pub mod object {
    use std::sync::RwLock;

    use arrow::datatypes::ArrowDataType;

    static OBJECT_PHYSICAL_DTYPE: RwLock<Option<ArrowDataType>> = RwLock::new(None);

    /// Records the type an object's values are laid out as, which `polars-core` does when the
    /// object registry is set.
    pub fn set_object_physical_type(dtype: ArrowDataType) {
        *OBJECT_PHYSICAL_DTYPE.write().unwrap() = Some(dtype);
    }

    /// # Panics
    /// Panics if no object type has been registered, as an object column cannot exist before one
    /// has been.
    #[cold]
    pub fn get_object_physical_type() -> ArrowDataType {
        OBJECT_PHYSICAL_DTYPE
            .read()
            .unwrap()
            .clone()
            .expect("no object type has been registered")
    }
}

/// Whether `dtype` is the `Struct {key, value}` a [`Map`](DataType::Map)'s entries are held as.
///
/// A map is stored as a list of two-field structs, so the type of those structs is part of what
/// makes a map well-formed — which is a question about types alone, and is asked wherever one is
/// built. The fields have to be *named*, which is what this asks: Arrow and Parquet define map
/// entries positionally, so only what comes from outside those two is held to it.
#[cfg(feature = "dtype-map")]
pub fn ensure_map_entries_dtype(dtype: &DataType) -> polars_error::PolarsResult<()> {
    use arrow::array::{MAP_KEY_NAME, MAP_VALUE_NAME};
    use polars_error::{polars_bail, polars_ensure};
    use polars_utils::pl_str::PlSmallStr;

    let DataType::Struct(fields) = dtype else {
        polars_bail!(InvalidOperation: "Map entries must be `Struct {{key, value}}`, got `{dtype}`")
    };
    // Spell the names out because `Struct` display abbreviates them to `struct[n]`.
    let mut names: Vec<&PlSmallStr> = fields.iter().map(|f| f.name()).collect();
    names.sort();
    polars_ensure!(
        names == [&MAP_KEY_NAME, &MAP_VALUE_NAME],
        InvalidOperation:
        "Map entries must be exactly two fields named `{}` and `{}`, got [{}]",
        MAP_KEY_NAME, MAP_VALUE_NAME,
        fields.iter().map(|f| format!("`{}`", f.name())).collect::<Vec<_>>().join(", "),
    );
    Ok(())
}
