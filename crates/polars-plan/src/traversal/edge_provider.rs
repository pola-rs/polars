use polars_utils::collection::{Collection, CollectionWrap};

pub trait NodeEdgesProvider<Edge: ?Sized> {
    fn inputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a;

    fn outputs<'a>(&'a mut self) -> CollectionWrap<Edge, &'a mut dyn Collection<Edge>>
    where
        Edge: 'a;

    /// # Panics
    /// Panics if indices are out of bounds
    fn swap_input_output(&mut self, input_idx: usize, output_idx: usize);

    /// # Panics
    /// Panics if indices are out of bounds
    fn get_input_output_mut(&mut self, input_idx: usize, output_idx: usize) -> [&mut Edge; 2];
}
