use std::ops::{Deref, DerefMut};

use polars_utils::arena::Node;
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey};

/// The physical plan while it is being built from the IR.
///
/// Every physical node is inserted through [`PhysSmBuilder::insert`], which records the IR
/// node whose lowering created it. Reads and in-place edits go through `Deref` to the slotmap.
pub struct PhysSmBuilder {
    phys_sm: SlotMap<PhysNodeKey, PhysNode>,
    /// IR node that nodes inserted through [`Self::insert`] are attributed to. Only ever set
    /// inside [`Self::with_ir_node`], so it is `None` between lowering scopes.
    current_ir_node: Option<Node>,
    /// Length of the IR arena before lowering started. Lowering appends temporary IR nodes
    /// beyond this index. They are not part of the plan the query observer sees, so nodes
    /// lowered from them inherit the attribution of the original IR node being lowered.
    original_ir_len: usize,
}

impl PhysSmBuilder {
    pub fn new(phys_sm: SlotMap<PhysNodeKey, PhysNode>, original_ir_len: usize) -> Self {
        Self {
            phys_sm,
            current_ir_node: None,
            original_ir_len,
        }
    }

    /// Whether `node` was part of the IR plan before lowering started.
    pub fn is_original_ir_node(&self, node: Node) -> bool {
        node.0 < self.original_ir_len
    }

    /// Runs `f` with `ir_node` as the attribution for every node inserted inside it, then
    /// restores the previous attribution.
    pub fn with_ir_node<R>(&mut self, ir_node: Node, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev = self.current_ir_node.replace(ir_node);
        let out = f(self);
        self.current_ir_node = prev;
        out
    }

    /// Inserts `node`. A node that already carries an attribution (a clone of a lowered node)
    /// keeps it; any other node is attributed to the enclosing [`Self::with_ir_node`] scope.
    ///
    /// This shadows `SlotMap::insert` through `Deref`: inherent methods win during method
    /// resolution, so `phys_sm.insert(..)` on a builder always stamps.
    ///
    /// # Panics
    /// With debug assertions, panics when the node ends up unattributed, which means it was
    /// inserted outside any IR node's lowering scope.
    pub fn insert(&mut self, mut node: PhysNode) -> PhysNodeKey {
        node.ir_node = node.ir_node.or(self.current_ir_node);
        debug_assert!(
            node.ir_node.is_some(),
            "physical node inserted outside of an IR node's lowering scope"
        );
        self.phys_sm.insert(node)
    }

    pub fn into_inner(self) -> SlotMap<PhysNodeKey, PhysNode> {
        self.phys_sm
    }
}

impl Deref for PhysSmBuilder {
    type Target = SlotMap<PhysNodeKey, PhysNode>;

    fn deref(&self) -> &Self::Target {
        &self.phys_sm
    }
}

impl DerefMut for PhysSmBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.phys_sm
    }
}
