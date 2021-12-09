pub(crate) trait IndexToUsize {
    /// Translate the negative index to an offset.
    fn to_usize(self, length: usize) -> Option<usize>;
}

impl IndexToUsize for i64 {
    fn to_usize(self, length: usize) -> Option<usize> {
        if self >= 0 && (self as usize) < length {
            Some(self as usize)
        } else {
            let subtract = self.abs() as usize;
            if subtract > length {
                None
            } else {
                Some(length - subtract)
            }
        }
    }
}
