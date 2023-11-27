
trait Idx: Copy {
    fn to_usize(self) -> usize;

    fn from_usize(i: usize) -> Self;
}

impl Idx for u32 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(i: usize) -> Self {
        i as u32
    }
}


impl Idx for u64 {
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(i: usize) -> Self {
        i as u64
    }
}

struct DoublyLinkedList<I: Idx> {
    r_link: Vec<I>,
    l_link: Vec<I>,
}

impl<I: Idx> DoublyLinkedList<I> {
    fn get_links(&mut self) -> (&mut [I], &mut [I]) {
        (&mut *self.l_link, &mut *self.r_link)
    }

    #[inline]
    fn len(&self) -> usize {
        self.l_link.len()
    }

    fn delete(&mut self, i: usize) {
        assert!(i < self.len());
        unsafe { self.delete_unchecked(i) }
    }

    unsafe fn delete_unchecked(&mut self, i: usize) {
        // left node
        let l = *self.l_link.get_unchecked(i);
        // right node
        let r = *self.r_link.get_unchecked(i);

        // connect rhs to other nodes lhs
        *self.r_link.get_unchecked_mut(l.to_usize()) = r;
        // connect lhs to other nodes rhs
        *self.l_link.get_unchecked_mut(r.to_usize()) = l
    }

    fn insert(&mut self, i: usize) {
        assert!(i < self.len());
        unsafe { self.insert_unchecked(i) }
    }

    unsafe fn insert_unchecked(&mut self, i: usize) {
        // left node
        let l = *self.l_link.get_unchecked(i);
        // right node
        let r = *self.r_link.get_unchecked(i);

        *self.r_link.get_unchecked_mut(l.to_usize()) = I::from_usize(i);
        *self.l_link.get_unchecked_mut(r.to_usize()) = I::from_usize(i);
    }
}