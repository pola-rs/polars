use crate::prelude::*;

impl From<StructChunked> for DataFrame {
    fn from(ca: StructChunked) -> Self {
        #[cfg(feature = "object")]
        {
            DataFrame::new_no_checks(ca.fields.clone())
        }
        #[cfg(not(feature = "object"))]
        {
            DataFrame::new_no_checks(ca.fields)
        }
    }
}

impl DataFrame {
    pub fn into_struct(self, name: &str) -> StructChunked {
        StructChunked::new(name, &self.columns).unwrap()
    }
}
