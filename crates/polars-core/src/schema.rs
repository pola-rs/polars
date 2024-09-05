use std::fmt::Debug;

use polars_utils::pl_str::PlSmallStr;

use crate::prelude::*;
use crate::utils::try_get_supertype;

pub type SchemaRef = Arc<Schema>;
pub type Schema = polars_schema::Schema<DataType>;

pub trait SchemaExt {
    fn from_arrow_schema(value: &ArrowSchema) -> Self;

    fn get_field(&self, name: &str) -> Option<Field>;

    fn try_get_field(&self, name: &str) -> PolarsResult<Field>;

    fn to_arrow(&self, compat_level: CompatLevel) -> ArrowSchema;

    fn iter_fields(&self) -> impl ExactSizeIterator<Item = Field> + '_;

    fn to_supertype(&mut self, other: &Schema) -> PolarsResult<bool>;
}

impl SchemaExt for Schema {
    fn from_arrow_schema(value: &ArrowSchema) -> Self {
        value
            .iter_values()
            .map(|x| (x.name.clone(), DataType::from_arrow(&x.data_type, true)))
            .collect()
    }

    /// Look up the name in the schema and return an owned [`Field`] by cloning the data.
    ///
    /// Returns `None` if the field does not exist.
    ///
    /// This method constructs the `Field` by cloning the name and dtype. For a version that returns references, see
    /// [`get`][Self::get] or [`get_full`][Self::get_full].
    fn get_field(&self, name: &str) -> Option<Field> {
        self.get_full(name)
            .map(|(_, name, dtype)| Field::new(name.clone(), dtype.clone()))
    }

    /// Look up the name in the schema and return an owned [`Field`] by cloning the data.
    ///
    /// Returns `Err(PolarsErr)` if the field does not exist.
    ///
    /// This method constructs the `Field` by cloning the name and dtype. For a version that returns references, see
    /// [`get`][Self::get] or [`get_full`][Self::get_full].
    fn try_get_field(&self, name: &str) -> PolarsResult<Field> {
        self.get_full(name)
            .ok_or_else(|| polars_err!(SchemaFieldNotFound: "{}", name))
            .map(|(_, name, dtype)| Field::new(name.clone(), dtype.clone()))
    }

    /// Convert self to `ArrowSchema` by cloning the fields.
    fn to_arrow(&self, compat_level: CompatLevel) -> ArrowSchema {
        self.iter()
            .map(|(name, dtype)| {
                (
                    name.clone(),
                    dtype.to_arrow_field(name.clone(), compat_level),
                )
            })
            .collect()
    }

    /// Iterates the [`Field`]s in this schema, constructing them anew by cloning each `(&name, &dtype)` pair.
    ///
    /// Note that this clones each name and dtype in order to form an owned [`Field`]. For a clone-free version, use
    /// [`iter`][Self::iter], which returns `(&name, &dtype)`.
    fn iter_fields(&self) -> impl ExactSizeIterator<Item = Field> + '_ {
        self.iter()
            .map(|(name, dtype)| Field::new(name.clone(), dtype.clone()))
    }

    /// Take another [`Schema`] and try to find the supertypes between them.
    fn to_supertype(&mut self, other: &Schema) -> PolarsResult<bool> {
        polars_ensure!(self.len() == other.len(), ComputeError: "schema lengths differ");

        let mut changed = false;
        for ((k, dt), (other_k, other_dt)) in self.iter_mut().zip(other.iter()) {
            polars_ensure!(k == other_k, ComputeError: "schema names differ: got {}, expected {}", k, other_k);

            let st = try_get_supertype(dt, other_dt)?;
            changed |= (&st != dt) || (&st != other_dt);
            *dt = st
        }
        Ok(changed)
    }
}

/// This trait exists to be unify the API of polars Schema and arrows Schema.
pub trait IndexOfSchema: Debug {
    /// Get the index of a column by name.
    fn index_of(&self, name: &str) -> Option<usize>;

    /// Get a vector of all column names.
    fn get_names(&self) -> Vec<&PlSmallStr>;

    fn get_names_str(&self) -> Vec<&str>;

    fn get_names_owned(&self) -> Vec<PlSmallStr>;

    fn try_index_of(&self, name: &str) -> PolarsResult<usize> {
        self.index_of(name).ok_or_else(|| {
            polars_err!(
                ColumnNotFound:
                "unable to find column {:?}; valid columns: {:?}", name, self.get_names(),
            )
        })
    }
}

impl IndexOfSchema for Schema {
    fn index_of(&self, name: &str) -> Option<usize> {
        self.index_of(name)
    }

    fn get_names(&self) -> Vec<&PlSmallStr> {
        self.iter_names().collect()
    }

    fn get_names_owned(&self) -> Vec<PlSmallStr> {
        self.iter_names().cloned().collect()
    }

    fn get_names_str(&self) -> Vec<&str> {
        self.iter_names().map(|x| x.as_str()).collect()
    }
}

impl IndexOfSchema for ArrowSchema {
    fn index_of(&self, name: &str) -> Option<usize> {
        self.iter_values().position(|f| f.name.as_str() == name)
    }

    fn get_names(&self) -> Vec<&PlSmallStr> {
        self.iter_values().map(|f| &f.name).collect()
    }

    fn get_names_owned(&self) -> Vec<PlSmallStr> {
        self.iter_values().map(|f| f.name.clone()).collect()
    }

    fn get_names_str(&self) -> Vec<&str> {
        self.iter_values().map(|f| f.name.as_str()).collect()
    }
}

pub trait SchemaNamesAndDtypes {
    const IS_ARROW: bool;
    type DataType: Debug + PartialEq;

    /// Get a vector of (name, dtype) pairs
    fn get_names_and_dtypes(&'_ self) -> Vec<(&'_ str, Self::DataType)>;
}

impl SchemaNamesAndDtypes for Schema {
    const IS_ARROW: bool = false;
    type DataType = DataType;

    fn get_names_and_dtypes(&'_ self) -> Vec<(&'_ str, Self::DataType)> {
        self.iter()
            .map(|(name, dtype)| (name.as_str(), dtype.clone()))
            .collect()
    }
}

impl SchemaNamesAndDtypes for ArrowSchema {
    const IS_ARROW: bool = true;
    type DataType = ArrowDataType;

    fn get_names_and_dtypes(&'_ self) -> Vec<(&'_ str, Self::DataType)> {
        self.iter_values()
            .map(|x| (x.name.as_str(), x.data_type.clone()))
            .collect()
    }
}

pub fn ensure_matching_schema<S: SchemaNamesAndDtypes>(lhs: &S, rhs: &S) -> PolarsResult<()> {
    let lhs = lhs.get_names_and_dtypes();
    let rhs = rhs.get_names_and_dtypes();

    if lhs.len() != rhs.len() {
        polars_bail!(
            SchemaMismatch:
            "schemas contained differing number of columns: {} != {}",
            lhs.len(), rhs.len(),
        );
    }

    for (i, ((l_name, l_dtype), (r_name, r_dtype))) in lhs.iter().zip(&rhs).enumerate() {
        if l_name != r_name {
            polars_bail!(
                SchemaMismatch:
                "schema names differ at index {}: {} != {}",
                i, l_name, r_name
            )
        }
        if l_dtype != r_dtype
            && (!S::IS_ARROW
                || unsafe {
                    // For timezone normalization. Easier than writing out the entire PartialEq.
                    DataType::from_arrow(
                        std::mem::transmute::<&<S as SchemaNamesAndDtypes>::DataType, &ArrowDataType>(
                            l_dtype,
                        ),
                        true,
                    ) != DataType::from_arrow(
                        std::mem::transmute::<&<S as SchemaNamesAndDtypes>::DataType, &ArrowDataType>(
                            r_dtype,
                        ),
                        true,
                    )
                })
        {
            polars_bail!(
                SchemaMismatch:
                "schema dtypes differ at index {} for column {}: {:?} != {:?}",
                i, l_name, l_dtype, r_dtype
            )
        }
    }

    Ok(())
}
