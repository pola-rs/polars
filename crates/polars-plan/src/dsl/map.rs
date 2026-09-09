use super::*;

/// Specialized expressions for [`DataType::Map`].
pub struct MapNameSpace(pub(crate) Expr);

impl MapNameSpace {
    /// Convert this `Map` to a `List` of `Struct {key, value}` entries.
    pub fn entries(self) -> Expr {
        self.0.map_unary(MapFunction::Entries)
    }
}
