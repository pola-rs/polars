use polars_core::prelude::DataType;
use super::*;

// struct MinReduce {
//     state: Scalar,
// }
//
// impl MinReduce {
//     fn new(dtype: DataType) -> Self {
//         Self {
//             state: Scalar::zero(&dtype),
//         }
//     }
// }
//
// impl Reduction for MinReduce {
//     fn init(&mut self) {
//         self.state = AnyValue::zero(&self.state.dtype());
//     }
//
//     fn update(&mut self, batch: &Series) {
//         batch.sum_as_series()
//         todo!()
//     }
//
//     fn combine(&mut self, other: &dyn Reduction) {
//         todo!()
//     }
//
//     fn finalize(&mut self) -> Scalar {
//         todo!()
//     }
// }