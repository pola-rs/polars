use crate::prelude::*;
pub mod logical_plan;
mod utils;

// impl ALogicalPlan {
//     pub(crate) fn input_node( self, arena: &Arena<ALogicalPlan>) -> Vec<Node> {
//         use ALogicalPlan::*;
//         match self {
//             Cache { input } => Some(input),
//             Sort { input, .. } => Some(input),
//             Explode { input, .. } => Some(input),
//             Selection { input, .. } => Some(input),
//             Projection { input, .. } => Some(input),
//             LocalProjection { input, .. } => schema,
//             Aggregate { schema, .. } => schema,
//             Join { l, .. } => schema,
//             HStack { schema, .. } => schema,
//             Distinct { input, .. } => arena.get(*input).schema(arena),
//             Slice { input, .. } => arena.get(*input).schema(arena),
//             Melt { schema, .. } => schema,
//             Udf { input, schema, .. } => match schema {
//                 Some(schema) => schema,
//                 None => arena.get(*input).schema(arena),
//             },
//             _ => None
//         }
//     }
// }
