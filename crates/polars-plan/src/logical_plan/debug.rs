#![allow(dead_code)]
use polars_utils::arena::{Arena, Node};

use crate::prelude::{node_to_expr, AExpr};

pub fn dbg_nodes(nodes: &[Node], arena: &Arena<AExpr>) {
    println!("[");
    for node in nodes {
        let e = node_to_expr(*node, arena);
        println!("{e:?}")
    }
    println!("]");
}
