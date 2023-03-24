use std::collections::BTreeSet;
use std::fmt::Debug;

use polars_core::prelude::*;
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::*;

/// Represents a pipeline/ branch in a subquery tree
#[derive(Default, Debug, Clone)]
pub(super) struct Branch {
    pub(super) streamable: bool,
    pub(super) sources: Vec<Node>,
    // joins seen in whole branch (we count a union as joins with multiple counts)
    pub(super) join_count: IdxSize,
    // node is operator/sink
    pub(super) operators_sinks: Vec<(IsSink, IsRhsJoin, Node)>,
}

/// Represents a subquery tree of pipelines.
type Tree<'a> = &'a [Branch];

/// We validate a tree in order to check if it is eligible for streaming.
/// It could be that a join branch wasn't added during collection of branches
/// (because it contained a non-streamable node). This function checks if every join
/// node has a match.
pub(super) fn is_valid_tree(tree: Tree) -> bool {
    if tree.is_empty() {
        return false;
    };

    // rhs joins will initially be placeholders
    let mut left_joins = BTreeSet::new();
    for branch in tree {
        for (_sink, is_rhs_join, node) in &branch.operators_sinks {
            if !*is_rhs_join {
                left_joins.insert(node.0);
            }
        }
    }
    for branch in tree {
        for (_sink, is_rhs_join, node) in &branch.operators_sinks {
            // check if every rhs join has a lhs join node
            if *is_rhs_join && !left_joins.contains(&node.0) {
                return false;
            }
        }
    }
    true
}

#[cfg(debug_assertions)]
#[allow(unused)]
pub(super) fn dbg_branch(b: &Branch, lp_arena: &Arena<ALogicalPlan>) {
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

    for (_, rhs_join, node) in &b.operators_sinks {
        let lp = lp_arena.get(*node);
        if *rhs_join {
            print!("rhs_join_placeholder -> ");
        } else {
            print!("{} -> ", lp.name());
        }
    }
    println!();
}

#[cfg(debug_assertions)]
#[allow(unused)]
pub(super) fn dbg_tree(tree: Tree, lp_arena: &Arena<ALogicalPlan>, expr_arena: &Arena<AExpr>) {
    if tree.is_empty() {
        println!("EMPTY TREE");
        return;
    }
    let root = tree
        .iter()
        .map(|branch| {
            let (_, _, root) = branch.operators_sinks.last().unwrap();
            *root
        })
        .max_by_key(|root| {
            // count the children of this root
            // the branch with the most children is the root of the whole tree
            lp_arena.iter(*root).count()
        })
        .unwrap();

    println!("SUBPLAN ELIGIBLE FOR STREAMING:");
    let lp = node_to_lp(root, expr_arena, &mut (lp_arena.clone()));
    println!("{lp:?}\n");

    println!("PIPELINE TREE:");
    for (i, branch) in tree.iter().enumerate() {
        print!("{i}: ");
        dbg_branch(branch, lp_arena);
    }
}
