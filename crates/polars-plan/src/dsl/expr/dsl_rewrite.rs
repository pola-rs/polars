use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use polars_core::prelude::*;

use crate::dsl::{Expr, SpecialEq};

/// A type of expression that can be rewritten into another expressions.
/// Usable in Plugins, so that they can write expressions based on metadata / type information.
pub trait DslRewrite: Send + Sync {
    /// What expression we resolve this into.
    /// Returns a template where `Expr::RewriteInput(i)` refers to input `i`.
    ///
    /// Polars does not validate the output type of the returned expression.
    /// plugin authors are responsible for testing that their rewrite produces the intended dtype.
    fn rewrite(&self, inputs: &[Field], input_schema: &Schema) -> PolarsResult<Expr>;

    /// Name used for display / error messages / hashing.
    fn name(&self) -> PlSmallStr;
}

/// Where the [`DslRewrite`] of a `FunctionExpr::DslRewrite` comes from.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DslRewriteSource {
    /// In-process Rust implementation. Cannot be serialized.
    Rust(SpecialEq<Arc<dyn DslRewrite>>),
}

impl DslRewriteSource {
    pub fn rewrite(&self, inputs: &[Field], input_schema: &Schema) -> PolarsResult<Expr> {
        match self {
            Self::Rust(r) => r.rewrite(inputs, input_schema),
        }
    }

    pub fn name(&self) -> PlSmallStr {
        match self {
            Self::Rust(r) => r.name(),
        }
    }
}

impl Hash for DslRewriteSource {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

impl Display for DslRewriteSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for SpecialEq<Arc<dyn DslRewrite>> {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        Err(S::Error::custom(format!(
            "cannot serialize in-process Rust rewrite '{}'",
            self.name()
        )))
    }
}

#[cfg(feature = "serde")]
impl<'a> serde::Deserialize<'a> for SpecialEq<Arc<dyn DslRewrite>> {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'a>,
    {
        use serde::de::Error;
        Err(D::Error::custom(
            "cannot deserialize in-process Rust rewrite",
        ))
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for SpecialEq<Arc<dyn DslRewrite>> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "DslRewrite".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "DslRewrite"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<u8>::json_schema(generator)
    }
}
