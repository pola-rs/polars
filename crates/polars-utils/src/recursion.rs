const STACK_SIZE_GUARANTEE: usize = 256 * 1024;
const STACK_ALLOC_SIZE: usize = 2 * 1024 * 1024;

pub fn with_dynamic_stack<R, F: FnOnce() -> R>(f: F) -> R {
    stacker::maybe_grow(STACK_SIZE_GUARANTEE, STACK_ALLOC_SIZE, f)
}
