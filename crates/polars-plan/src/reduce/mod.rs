use polars_core::datatypes::Scalar;
use polars_core::prelude::Series;

#[allow(dead_code)]
trait Reduction {
    fn init(&mut self);

    fn update(&mut self, batch: &Series);

    fn combine(&mut self, other: &dyn Reduction);

    fn finalize(&mut self) -> Scalar;
}
