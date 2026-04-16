/// Vec container with a getter that clears the vec.
#[derive(Default)]
pub struct ScratchVec<T>(Vec<T>);

impl<T> ScratchVec<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Clear the vec and return a mutable reference to it.
    pub fn get(&mut self) -> &mut Vec<T> {
        self.0.clear();
        &mut self.0
    }
}
