pub struct Pool;

impl Pool {
    pub fn current_num_threads(&self) -> usize {
        rayon::current_num_threads()
    }

    pub fn current_thread_index(&self) -> Option<usize> {
        rayon::current_thread_index()
    }

    pub fn current_thread_has_pending_tasks(&self) -> Option<bool> {
        None
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

    pub fn scope<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&rayon::Scope<'scope>) -> R + Send,
        R: Send,
    {
        rayon::scope(op)
    }
}
