use std::collections::BTreeSet;
use std::fmt::Debug;

use polars_plan::prelude::*;

#[derive(Copy, Clone, Debug)]
pub(super) enum PipelineNode {
    Sink(Node),
    Operator(Node),
    RhsJoin(Node),
    Union(Node),
}

impl PipelineNode {
    pub(super) fn node(self) -> Node {
        match self {
            Self::Sink(node) => node,
            Self::Operator(node) => node,
            Self::RhsJoin(node) => node,
            Self::Union(node) => node,
        }
    }
}

/// Represents a pipeline/ branch in a subquery tree
#[derive(Default, Debug, Clone)]
pub(super) struct Branch {
    // During traversal of ALP
    // we determine the execution order
    // as traversal order == execution order
    // we can increment this counter
    // the individual branches are then flattened
    // sorted and executed in reversed order
    // (to traverse from leaves to root)
    pub(super) execution_id: u32,
    pub(super) streamable: bool,
    pub(super) sources: Vec<Node>,
    // joins seen in whole branch (we count a union as joins with multiple counts)
    pub(super) join_count: u32,
    // node is operator/sink
    pub(super) operators_sinks: Vec<PipelineNode>,
}

fn sink_node(pl_node: &PipelineNode) -> Option<Node> {
    match pl_node {
        PipelineNode::Sink(node) => Some(*node),
        _ => None,
    }
}

impl Branch {
    pub(super) fn get_final_sink(&self) -> Option<Node> {
        // this is still in the order of discovery
        // so the first sink is the final one.
        self.operators_sinks.iter().find_map(sink_node)
    }
    pub(super) fn split(&self) -> Self {
        Self {
            execution_id: self.execution_id,
            streamable: self.streamable,
            join_count: self.join_count,
            ..Default::default()
        }
    }

    /// this will share the sink
    pub(super) fn split_from_sink(&self) -> Self {
        match self
            .operators_sinks
            .iter()
            .rposition(|pl_node| sink_node(pl_node).is_some())
        {
            None => self.split(),
            Some(pos) => Self {
                execution_id: self.execution_id,
                streamable: self.streamable,
                join_count: self.join_count,
                operators_sinks: self.operators_sinks[pos..].to_vec(),
                ..Default::default()
            },
        }
    }
}

/// Represents a subquery tree of pipelines.
type TreeRef<'a> = &'a [Branch];
pub(super) type Tree = Vec<Branch>;

/// We validate a tree in order to check if it is eligible for streaming.
/// It could be that a join branch wasn't added during collection of branches
/// (because it contained a non-streamable node). This function checks if every join
/// node has a match.
pub(super) fn is_valid_tree(tree: TreeRef) -> bool {
    if tree.is_empty() {
        return false;
    };
    let joins_in_tree = tree.iter().map(|branch| branch.join_count).sum::<u32>();
    let branches_in_tree = tree.len() as u32;

    // all join branches should be added, if not we skip the tree, as it is invalid
    if (branches_in_tree - 1) != joins_in_tree {
        return false;
    }

    // rhs joins will initially be placeholders
    let mut left_joins = BTreeSet::new();
    for branch in tree {
        for pl_node in &branch.operators_sinks {
            if !matches!(pl_node, PipelineNode::RhsJoin(_)) {
                left_joins.insert(pl_node.node().0);
            }
        }
    }
    for branch in tree {
        for pl_node in &branch.operators_sinks {
            // check if every rhs join has a lhs join node
            if matches!(pl_node, PipelineNode::RhsJoin(_))
                && !left_joins.contains(&pl_node.node().0)
            {
                return false;
            }
        }
    }
    true
}

#[cfg(debug_assertions)]
#[allow(unused)]
pub(super) fn dbg_branch(b: &Branch, lp_arena: &Arena<IR>) {
    // streamable: bool,
    // sources: Vec<Node>,
    // // joins seen in whole branch (we count a union as joins with multiple counts)
    // join_count: IdxSize,
    // // node is operator/sink
    // operators_sinks: Vec<(IsSink, IsRhsJoin, Node)>,

    if b.streamable {
        print!("streamable: ")
    } else {
        print!("non-streamable: ")
    }
    for src in &b.sources {
        let lp = lp_arena.get(*src);
        print!("{}, ", lp.name());
    }
    print!("=> ");

    for pl_node in &b.operators_sinks {
        let lp = lp_arena.get(pl_node.node());
        if matches!(pl_node, PipelineNode::RhsJoin(_)) {
            print!("rhs_join_placeholder -> ");
        } else {
            print!("{} -> ", lp.name());
        }
    }
    println!();
}

#[cfg(debug_assertions)]
#[allow(unused)]
pub(super) fn dbg_tree(tree: Tree, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) {
    if tree.is_empty() {
        println!("EMPTY TREE");
        return;
    }
    let root = tree
        .iter()
        .map(|branch| {
            let pl_node = branch.operators_sinks.last().unwrap();
            pl_node.node()
        })
        .max_by_key(|root| {
            // count the children of this root
            // the branch with the most children is the root of the whole tree
            lp_arena.iter(*root).count()
        })
        .unwrap();

    println!("SUBPLAN ELIGIBLE FOR STREAMING:");
    println!(
        "{}\n",
        IRPlanRef {
            lp_top: root,
            lp_arena,
            expr_arena
        }
        .display()
    );

    println!("PIPELINE TREE:");
    for (i, branch) in tree.iter().enumerate() {
        print!("{i}: ");
        dbg_branch(branch, lp_arena);
    }
}
