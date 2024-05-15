mod extrema;

use polars_core::prelude::{AnyValue, DataType, Series};

type Scalar = AnyValue<'static>;

trait Reduction {
    fn init(&mut self);

    fn update(&mut self, batch: &Series);

    fn combine(&mut self, other: &dyn Reduction);

    fn finalize(&mut self) -> Scalar;
}
