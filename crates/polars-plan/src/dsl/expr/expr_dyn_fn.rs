use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::Arc;

use super::*;

pub trait AnonymousColumnsUdf: ColumnsUdf {
    fn as_column_udf(self: Arc<Self>) -> Arc<dyn ColumnsUdf>;
    fn deep_clone(self: Arc<Self>) -> Arc<dyn AnonymousColumnsUdf>;

    fn try_serialize(&self, _buf: &mut Vec<u8>) -> PolarsResult<()> {
        polars_bail!(ComputeError: "serialization not supported for this 'opaque' function")
    }

    fn get_field(&self, input_schema: &Schema, fields: &[Field]) -> PolarsResult<Field>;
}

/// A wrapper trait for any closure `Fn(Vec<Series>) -> PolarsResult<Series>`
pub trait ColumnsUdf: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any {
        unimplemented!("as_any not implemented for this 'opaque' function")
    }

    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Column>;
}

impl<F> ColumnsUdf for F
where
    F: Fn(&mut [Column]) -> PolarsResult<Column> + Send + Sync,
{
    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Column> {
        self(s)
    }
}

impl Debug for dyn ColumnsUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ColumnUdf")
    }
}

#[derive(Clone)]
/// Wrapper type that has special equality properties
/// depending on the inner type specialization
pub struct SpecialEq<T>(T);

impl<T> SpecialEq<T> {
    pub fn new(val: T) -> Self {
        SpecialEq(val)
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl SpecialEq<Arc<dyn AnonymousColumnsUdf>> {
    pub fn deep_clone(self) -> Self {
        SpecialEq(self.0.deep_clone())
    }
}

impl<T: ?Sized> PartialEq for SpecialEq<Arc<T>> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T: ?Sized> Eq for SpecialEq<Arc<T>> {}

impl<T: ?Sized> Hash for SpecialEq<Arc<T>> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(self).hash(state);
    }
}

impl PartialEq for SpecialEq<Series> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Debug for SpecialEq<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "no_eq")
    }
}

impl<T> Deref for SpecialEq<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct BaseColumnUdf<F, DT> {
    f: F,
    dt: DT,
}

impl<F, DT> BaseColumnUdf<F, DT> {
    pub fn new(f: F, dt: DT) -> Self {
        Self { f, dt }
    }
}

impl<F, DT> ColumnsUdf for BaseColumnUdf<F, DT>
where
    F: Fn(&mut [Column]) -> PolarsResult<Column> + Send + Sync,
    DT: Fn(&Schema, &[Field]) -> PolarsResult<Field> + Send + Sync,
{
    fn call_udf(&self, s: &mut [Column]) -> PolarsResult<Column> {
        (self.f)(s)
    }
}

impl<F, DT> AnonymousColumnsUdf for BaseColumnUdf<F, DT>
where
    F: Fn(&mut [Column]) -> PolarsResult<Column> + 'static + Send + Sync,
    DT: Fn(&Schema, &[Field]) -> PolarsResult<Field> + 'static + Send + Sync,
{
    fn as_column_udf(self: Arc<Self>) -> Arc<dyn ColumnsUdf> {
        self as _
    }
    fn deep_clone(self: Arc<Self>) -> Arc<dyn AnonymousColumnsUdf> {
        self
    }

    fn get_field(&self, input_schema: &Schema, fields: &[Field]) -> PolarsResult<Field> {
        (self.dt)(input_schema, fields)
    }
}

pub type OpaqueColumnUdf = LazySerde<SpecialEq<Arc<dyn AnonymousColumnsUdf>>>;
pub(crate) fn new_column_udf<F: AnonymousColumnsUdf + 'static>(func: F) -> OpaqueColumnUdf {
    LazySerde::Deserialized(SpecialEq::new(Arc::new(func)))
}

impl OpaqueColumnUdf {
    pub fn materialize(self) -> PolarsResult<SpecialEq<Arc<dyn AnonymousColumnsUdf>>> {
        match self {
            Self::Deserialized(t) => Ok(t),
            Self::Named {
                name,
                payload,
                value,
            } => feature_gated!("serde", {
                use super::named_serde::NAMED_SERDE_REGISTRY_EXPR;
                match value {
                    Some(v) => Ok(v),
                    None => Ok(SpecialEq(
                        NAMED_SERDE_REGISTRY_EXPR
                            .read()
                            .unwrap()
                            .as_ref()
                            .expect("NAMED EXPR REGISTRY NOT SET")
                            .get_function(&name, payload.unwrap().as_ref())
                            .expect("NAMED FUNCTION NOT FOUND"),
                    )),
                }
            }),
            Self::Bytes(_b) => {
                feature_gated!("serde";"python", {
                    serde_expr::deserialize_column_udf(_b.as_ref()).map(SpecialEq::new)
                })
            },
        }
    }
}
