use polars_utils::vec::CapacityByFactor;

use super::*;
use crate::constants::CSE_REPLACED;
use crate::logical_plan::projection_expr::ProjectionExprs;
use crate::prelude::visitor::ALogicalPlanNode;

// // We use hashes to get an Identifier
// // but this is very hard to debug, so we also have a version that
// // uses a string trail.
// #[cfg(test)]
// mod identifier_impl {
//     use std::hash::{Hash, Hasher};
//
//     use super::*;
//     /// Identifier that shows the sub-expression path.
//     /// Must implement hash and equality and ideally
//     /// have little collisions
//     /// We will do a full expression comparison to check if the
//     /// expressions with equal identifiers are truly equal
//     #[derive(Clone, Debug)]
//     pub struct Identifier {
//         inner: String,
//         last_node: Option<ALogicalPlanNode>,
//     }
//
//     impl PartialEq for Identifier {
//         fn eq(&self, other: &Self) -> bool {
//             self.inner == other.inner && self.last_node == other.last_node
//         }
//     }
//
//     impl Eq for Identifier {}
//
//     impl Hash for Identifier {
//         fn hash<H: Hasher>(&self, state: &mut H) {
//             self.inner.hash(state)
//         }
//     }
//
//     impl Identifier {
//         pub fn new() -> Self {
//             Self {
//                 inner: String::new(),
//                 last_node: None,
//             }
//         }
//
//         pub fn alp_node(&self) -> ALogicalPlanNode {
//             self.last_node.unwrap()
//         }
//
//         pub fn is_valid(&self) -> bool {
//             !self.inner.is_empty()
//         }
//
//         pub fn materialize(&self) -> String {
//             format!("{}{}", CSE_REPLACED, self.inner)
//         }
//
//         pub fn combine(&mut self, other: &Identifier) {
//             self.inner.push('!');
//             self.inner.push_str(&other.inner);
//         }
//
//         pub fn add_alp_node(&self, ae: &ALogicalPlanNode) -> Self {
//             let inner = format!("{:E}{}", ae.to_alp(), self.inner);
//             Self {
//                 inner,
//                 last_node: Some(*ae),
//             }
//         }
//     }
// }
//
// #[cfg(not(test))]
// mod identifier_impl {
//     use std::hash::{Hash, Hasher};
//
//     use ahash::RandomState;
//     use polars_core::hashing::_boost_hash_combine;
//
//     use super::*;
//     /// Identifier that shows the sub-expression path.
//     /// Must implement hash and equality and ideally
//     /// have little collisions
//     /// We will do a full expression comparison to check if the
//     /// expressions with equal identifiers are truly equal
//     #[derive(Clone, Debug)]
//     pub struct Identifier {
//         inner: Option<u64>,
//         last_node: Option<ALogicalPlanNode>,
//         hb: RandomState,
//     }
//
//     impl PartialEq<Self> for Identifier {
//         fn eq(&self, other: &Self) -> bool {
//             self.inner == other.inner && self.last_node == other.last_node
//         }
//     }
//
//     impl Eq for Identifier {}
//
//     impl Hash for Identifier {
//         fn hash<H: Hasher>(&self, state: &mut H) {
//             state.write_u64(self.inner.unwrap_or(0))
//         }
//     }
//
//     impl Identifier {
//         pub fn new() -> Self {
//             Self {
//                 inner: None,
//                 last_node: None,
//                 hb: RandomState::with_seed(0),
//             }
//         }
//
//         pub fn alp_node(&self) -> ALogicalPlanNode {
//             self.last_node.unwrap()
//         }
//
//         pub fn is_valid(&self) -> bool {
//             self.inner.is_some()
//         }
//
//         pub fn materialize(&self) -> String {
//             format!("{}{}", CSE_REPLACED, self.inner.unwrap_or(0))
//         }
//
//         pub fn combine(&mut self, other: &Identifier) {
//             let inner = match (self.inner, other.inner) {
//                 (Some(l), Some(r)) => _boost_hash_combine(l, r),
//                 (None, Some(r)) => r,
//                 (Some(l), None) => l,
//                 _ => return,
//             };
//             self.inner = Some(inner);
//         }
//
//         pub fn add_alp_node(&self, alp: &ALogicalPlanNode) -> Self {
//             let hashed = self.hb.hash_one(alp.to_alp());
//             let inner = Some(
//                 self.inner
//                     .map_or(hashed, |l| _boost_hash_combine(l, hashed)),
//             );
//             Self {
//                 inner,
//                 last_node: Some(*alp),
//                 hb: self.hb.clone(),
//             }
//         }
//     }
// }
// use identifier_impl::*;
