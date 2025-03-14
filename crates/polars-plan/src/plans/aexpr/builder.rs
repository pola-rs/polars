use super::*;

pub struct AExprBuilder<'a> {
    arena: &'a mut Arena<AExpr>,
    node: Node,
}

impl<'a> AExprBuilder<'a> {
    pub fn new(node: Node, arena: &'a mut Arena<AExpr>) -> Self {
        Self { arena, node }
    }

    fn _add(mut self, ae: AExpr) -> Self {
        self.node = self.arena.add(ae);
        self
    }

    pub fn implode(self) -> Self {
        let agg = AExpr::Agg(IRAggExpr::Implode(self.node));
        self._add(agg)
    }

    pub fn cast(self, dtype: DataType, options: CastOptions) -> Self {
        let agg = AExpr::Cast {
            expr: self.node,
            dtype,
            options,
        };
        self._add(agg)
    }

    pub fn build_node(self) -> Node {
        self.node
    }

    pub fn build_ae(self) -> AExpr {
        if self.node.0 == self.arena.len() - 1 {
            self.arena.pop().unwrap()
        } else {
            self.arena.take(self.node)
        }
    }
}
