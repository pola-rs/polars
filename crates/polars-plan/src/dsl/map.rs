use super::*;

/// Specialized expressions for [`DataType::Map`].
pub struct MapNameSpace(pub(crate) Expr);

impl MapNameSpace {
    /// Convert this `Map` to a `List` of `Struct {key, value}` entries.
    ///
    /// Null maps remain null.
    pub fn entries(self) -> Expr {
        self.0.map_unary(MapFunction::Entries)
    }

    /// Get the keys of every map as a `List`, in entry order.
    ///
    /// Null maps remain null.
    pub fn keys(self) -> Expr {
        self.0.map_unary(MapFunction::Keys)
    }

    /// Get the values of every map as a `List`, in entry order.
    ///
    /// Null maps remain null.
    pub fn values(self) -> Expr {
        self.0.map_unary(MapFunction::Values)
    }

    /// Get the number of entries of every map.
    ///
    /// Null maps remain null.
    pub fn len(self) -> Expr {
        self.0.map_unary(MapFunction::Length)
    }

    /// Check whether every map holds `key`.
    ///
    /// Null maps remain null. Map keys are never null, so a null `key` is never found.
    pub fn contains_key(self, key: Expr) -> Expr {
        self.0.map_binary(MapFunction::ContainsKey, key)
    }

    /// Look up `key` in every map, returning the map's value dtype.
    ///
    /// Maps that do not hold `key` yield null, as do null maps. Map keys are never null, so a
    /// null `key` is never found.
    pub fn get(self, key: Expr) -> Expr {
        self.0.map_binary(MapFunction::Get, key)
    }
}
