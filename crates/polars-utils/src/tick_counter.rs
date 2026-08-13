/// **Unsynchronized** tick counter.
///
/// This is intended to be used to *roughly* organize events across threads in
/// their received order but without any actual guarantees around monotonicity
/// or synchronization, for maximum speed.
///
/// This does not guarantee monotonicity or any particular (fixed) time unit.
/// If correctness matters use Instant::now().
#[inline]
pub fn tick_counter() -> u64 {
    cfg_select! {
        target_arch = "x86_64" => {
            unsafe { core::arch::x86_64::_rdtsc() }
        }

        target_arch = "aarch64" => {
            let cnt: u64;
            unsafe {
                core::arch::asm!(
                    "mrs {cnt}, cntvct_el0",
                    cnt = out(reg) cnt,
                    options(nomem, nostack, preserves_flags),
                );
            }
            cnt
        }

        _ => {
            use std::sync::LazyLock;
            use std::time::Instant;

            static REFERENCE_INSTANT: LazyLock<Instant> = LazyLock::new(Instant::now);
            REFERENCE_INSTANT.elapsed().as_nanos() as u64
        }
    }
}
