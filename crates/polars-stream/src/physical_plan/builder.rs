use std::ops::{Deref, DerefMut};

use polars_utils::arena::Node;
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey};

/// The physical plan while it is being built from the IR.
///
/// Every physical node is inserted through [`PhysPlanBuilder::insert`], which records the IR
/// node whose lowering created it. Reads and in-place edits go through `Deref` to the slotmap.
pub struct PhysPlanBuilder {
    phys_sm: SlotMap<PhysNodeKey, PhysNode>,
    /// IR node that nodes inserted through [`Self::insert`] are attributed to.
    pub current_ir_node: Option<Node>,
    /// Length of the IR arena before lowering started. Lowering appends temporary IR nodes
    /// beyond this index. They are not part of the plan the query observer sees, so nodes
    /// lowered from them inherit the attribution of the original IR node being lowered.
    original_ir_len: usize,
}

impl PhysPlanBuilder {
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

    /// Inserts `node`, attributing it to [`Self::current_ir_node`].
    ///
    /// This shadows `SlotMap::insert` through `Deref`: inherent methods win during method
    /// resolution, so `phys_sm.insert(..)` on a builder always stamps.
    pub fn insert(&mut self, mut node: PhysNode) -> PhysNodeKey {
        node.ir_node = self.current_ir_node;
        self.phys_sm.insert(node)
    }

    pub fn into_inner(self) -> SlotMap<PhysNodeKey, PhysNode> {
        self.phys_sm
    }
}

impl Deref for PhysPlanBuilder {
    type Target = SlotMap<PhysNodeKey, PhysNode>;

    fn deref(&self) -> &Self::Target {
        &self.phys_sm
    }
}

impl DerefMut for PhysPlanBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.phys_sm
    }
}
