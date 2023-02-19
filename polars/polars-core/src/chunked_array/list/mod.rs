//! Special list utility methods
mod iterator;

use crate::chunked_array::Settings;
use crate::prelude::*;

impl ListChunked {
    #[cfg(feature = "private")]
    pub fn set_fast_explode(&mut self) {
        self.bit_settings.insert(Settings::FAST_EXPLODE_LIST)
    }
    pub(crate) fn unset_fast_explode(&mut self) {
        self.bit_settings.remove(Settings::FAST_EXPLODE_LIST)
    }

    pub fn _can_fast_explode(&self) -> bool {
        self.bit_settings.contains(Settings::FAST_EXPLODE_LIST)
    }

    pub(crate) fn is_nested(&self) -> bool {
        match self.dtype() {
            DataType::List(inner) => matches!(&**inner, DataType::List(_)),
            _ => unreachable!(),
        }
    }

    pub fn to_logical(&mut self, inner_dtype: DataType) {
        assert_eq!(inner_dtype.to_physical(), self.inner_dtype());
        let fld = Arc::make_mut(&mut self.field);
        fld.coerce(DataType::List(Box::new(inner_dtype)))
    }

    pub fn apply_to_inner(
        &self,
        func: &dyn Fn(Series) -> PolarsResult<Series>,
    ) -> PolarsResult<ListChunked> {
        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();
        let elements = Series::try_from(("", arr.values().clone())).unwrap();

        let expected_len = elements.len();
        let out: Series = func(elements)?;
        if out.len() != expected_len {
            return Err(PolarsError::ComputeError(
                "The function should apply only elementwise Instead it has removed elements".into(),
            ));
        }
        let out = out.rechunk();
        let values = out.chunks()[0].clone();

        let inner_dtype = LargeListArray::default_datatype(out.dtype().to_arrow());
        let arr = LargeListArray::new(
            inner_dtype,
            (*arr.offsets()).clone(),
            values,
            arr.validity().cloned(),
        );
        unsafe { Ok(ListChunked::from_chunks(self.name(), vec![Box::new(arr)])) }
    }
}
