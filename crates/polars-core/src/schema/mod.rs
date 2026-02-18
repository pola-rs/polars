use std::fmt::Debug;

use arrow::bitmap::Bitmap;
use polars_utils::pl_str::PlSmallStr;

use crate::prelude::*;
use crate::utils::try_get_supertype;

pub mod iceberg;

pub type SchemaRef = Arc<Schema>;
pub type Schema = polars_schema::Schema<DataType, ()>;

pub trait SchemaExt {
    fn from_arrow_schema(value: &ArrowSchema) -> Self;

    fn get_field(&self, name: &str) -> Option<Field>;

    fn try_get_field(&self, name: &str) -> PolarsResult<Field>;

    fn to_arrow(&self, compat_level: CompatLevel) -> ArrowSchema;

    fn iter_fields(&self) -> impl ExactSizeIterator<Item = Field> + '_;

    fn to_supertype(&mut self, other: &Schema) -> PolarsResult<bool>;

    /// Select fields using a bitmap.
    fn project_select(&self, select: &Bitmap) -> Self;
}

impl SchemaExt for Schema {
    fn from_arrow_schema(value: &ArrowSchema) -> Self {
        value
            .iter_values()
            .map(|x| (x.name.clone(), DataType::from_arrow_field(x)))
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
            .ok_or_else(|| polars_err!(SchemaFieldNotFound: "{name}"))
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

    fn project_select(&self, select: &Bitmap) -> Self {
        assert_eq!(self.len(), select.len());
        self.iter()
            .zip(select.iter())
            .filter(|(_, select)| *select)
            .map(|((n, dt), _)| (n.clone(), dt.clone()))
            .collect()
    }
}

pub trait SchemaNamesAndDtypes {
    const IS_ARROW: bool;
    type DataType: Debug + Clone + PartialEq;

    fn iter_names_and_dtypes(
        &self,
    ) -> impl ExactSizeIterator<Item = (&PlSmallStr, &Self::DataType)>;
}

impl SchemaNamesAndDtypes for ArrowSchema {
    const IS_ARROW: bool = true;
    type DataType = ArrowDataType;

    fn iter_names_and_dtypes(
        &self,
    ) -> impl ExactSizeIterator<Item = (&PlSmallStr, &Self::DataType)> {
        self.iter_values().map(|x| (&x.name, &x.dtype))
    }
}

impl SchemaNamesAndDtypes for Schema {
    const IS_ARROW: bool = false;
    type DataType = DataType;

    fn iter_names_and_dtypes(
        &self,
    ) -> impl ExactSizeIterator<Item = (&PlSmallStr, &Self::DataType)> {
        self.iter()
    }
}

pub fn ensure_matching_schema<F, M>(
    lhs: &polars_schema::Schema<F, M>,
    rhs: &polars_schema::Schema<F, M>,
) -> PolarsResult<()>
where
    polars_schema::Schema<F, M>: SchemaNamesAndDtypes,
{
    let lhs = lhs.iter_names_and_dtypes();
    let rhs = rhs.iter_names_and_dtypes();

    if lhs.len() != rhs.len() {
        polars_bail!(
            SchemaMismatch:
            "schemas contained differing number of columns: {} != {}",
            lhs.len(), rhs.len(),
        );
    }

    for (i, ((l_name, l_dtype), (r_name, r_dtype))) in lhs.zip(rhs).enumerate() {
        if l_name != r_name {
            polars_bail!(
                SchemaMismatch:
                "schema names differ at index {}: {} != {}",
                i, l_name, r_name
            )
        }
        if l_dtype != r_dtype
            && (!polars_schema::Schema::<F, M>::IS_ARROW
                || unsafe {
                    // For timezone normalization. Easier than writing out the entire PartialEq.
                    DataType::from_arrow_dtype(std::mem::transmute::<
                        &<polars_schema::Schema<F, M> as SchemaNamesAndDtypes>::DataType,
                        &ArrowDataType,
                    >(l_dtype))
                        != DataType::from_arrow_dtype(std::mem::transmute::<
                            &<polars_schema::Schema<F, M> as SchemaNamesAndDtypes>::DataType,
                            &ArrowDataType,
                        >(r_dtype))
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
