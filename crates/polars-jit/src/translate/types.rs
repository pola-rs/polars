use crate::compiler::ListBindings;
use crate::jit_program::JitProgram;

pub(crate) struct TranslationResult {
    pub(crate) program: JitProgram,
    pub(crate) bindings: ListBindings,
}
