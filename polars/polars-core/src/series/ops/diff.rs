use crate::prelude::*;
use crate::series::ops::NullBehavior;

impl Series {
    #[cfg_attr(docsrs, doc(cfg(feature = "diff")))]
    pub fn diff(&self, n: usize, null_behavior: NullBehavior) -> Series {
        match null_behavior {
            NullBehavior::Ignore => self - &self.shift(n as i64),
            NullBehavior::Drop => {
                let len = self.len() - n;
                &self.slice(n as i64, len) - &self.slice(0, len)
            }
        }
    }
}
