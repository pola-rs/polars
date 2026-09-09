use super::*;

/// Specialized expressions for [`DataType::Map`].
pub struct MapNameSpace(pub(crate) Expr);

impl MapNameSpace {
    /// Convert this `Map` to a `List` of `Struct {key, value}` entries.
    ///
    /// Null rows remain null, and entries retained by them are omitted.
    pub fn entries(self) -> Expr {
        self.0.map_unary(MapFunction::Entries)
    }
}
