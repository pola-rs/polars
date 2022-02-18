use super::*;

#[cfg(feature = "dtype-categorical")]
impl<'a> dyn SeriesTrait + 'a {
    pub(crate) fn as_mut_categorical(&mut self) -> &mut CategoricalChunked {
        if matches!(self.dtype(), DataType::Categorical(_)) {
            #[cfg(debug_assertions)]
            {
                self.as_any().downcast_ref::<CategoricalChunked>().unwrap();
            }

            unsafe { &mut *(self as *mut dyn SeriesTrait as *mut CategoricalChunked) }
        } else {
            panic!("implementation error")
        }
    }
}
