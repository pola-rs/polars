pub struct Pool;

impl Pool {
    pub fn current_num_threads(&self) -> usize {
        rayon::current_num_threads()
    }

    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        op()
    }

    pub fn join<A, B, RA, RB>(&self, oper_a: A, oper_b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        rayon::join(oper_a, oper_b)
    }

    pub fn spawn<F>(&self, func: F)
    where
        F: 'static + FnOnce() + Send,
    {
        rayon::spawn(func);
    }
}
