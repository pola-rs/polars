use cranelift_jit::JITModule;

use crate::jit_program::InputId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ListBindings {
    pub(crate) state_input: InputId,
    pub(crate) row_inputs: Box<[InputId]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CompilationShape {
    Lists(ListBindings),
}

pub(super) type ListFunction = unsafe extern "C" fn(*const *const u8, usize, *const u8, *mut u8);

pub struct CompiledProgram {
    pub(crate) function: ListFunction,
    pub(super) _module: OwnedModule,
}

pub(super) struct OwnedModule(Option<JITModule>);

impl OwnedModule {
    pub(super) fn new(module: JITModule) -> Self {
        Self(Some(module))
    }

    pub(super) fn module_mut(&mut self) -> &mut JITModule {
        self.0.as_mut().unwrap()
    }
}

impl Drop for OwnedModule {
    fn drop(&mut self) {
        if let Some(module) = self.0.take() {
            // SAFETY: CompiledProgram owns the module and its function pointer.
            unsafe { module.free_memory() };
        }
    }
}
