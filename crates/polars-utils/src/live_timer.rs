///
/// Live timer from Orson
///
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Counts for how much time this timer was 'live'.
///
/// It is live when there is at least one session, but multiple concurrent
/// sessions don't increase the rate at which the timer ticks.
pub struct LiveTimer {
    base_timestamp: Instant,
    live_sessions: AtomicU64,

    /// Nanoseconds since base_timestamp.
    current_segment_start: AtomicU64,
    /// Bottom bit indicates whether the current segment should be accounted for.
    accumulated_ns: AtomicU64,
}

impl LiveTimer {
    pub fn new() -> Self {
        Self {
            base_timestamp: Instant::now(),
            live_sessions: AtomicU64::new(0),
            current_segment_start: AtomicU64::new(0),
            accumulated_ns: AtomicU64::new(0),
        }
    }

    pub fn total_time_live_ns(&self) -> u64 {
        // Load accum and current_segment_start in a self-consistent way.
        let mut current_segment_start = 0;
        let mut accum = self.accumulated_ns.load(Ordering::Acquire); // This Acquire prevents cur_seg.load() from being reordered before.
        while accum & 1 != 0 {
            current_segment_start = self.current_segment_start.load(Ordering::Relaxed);
            let accum2 = self.accumulated_ns.fetch_add(0, Ordering::AcqRel); // This Release prevents cur_seg.load() from being reordered after.
            if accum == accum2 {
                break; // No change to accumulator, current_segment must be correct for it.
            } else {
                accum = accum2; // Accumulator changed, try again.
            }
        }

        let mut total_ns = accum >> 1;
        if accum & 1 != 0 {
            let start = self.base_timestamp + Duration::from_nanos(current_segment_start);
            total_ns += start.elapsed().as_nanos() as u64;
        }
        total_ns
    }

    // May not be concurrently called with exclusive_stop_current_segment.
    fn exclusive_start_current_segment(&self) {
        let start = Instant::now()
            .duration_since(self.base_timestamp)
            .as_nanos() as u64;
        self.current_segment_start.store(start, Ordering::Release);
        let accum = self.accumulated_ns.load(Ordering::Relaxed);
        self.accumulated_ns.store(accum | 1, Ordering::Release);
    }

    // May not be concurrently called with exclusive_start_current_segment.
    fn exclusive_stop_current_segment(&self) {
        let current_segment_start = self.current_segment_start.load(Ordering::Acquire);
        let start = self.base_timestamp + Duration::from_nanos(current_segment_start);
        let accum = self.accumulated_ns.load(Ordering::Relaxed);
        let new_accum = (accum & !1) + ((start.elapsed().as_nanos() as u64) << 1);
        self.accumulated_ns.store(new_accum, Ordering::Relaxed);
    }

    pub fn start_session(&self) {
        if self.live_sessions.fetch_add(1, Ordering::Acquire) == 0 {
            self.exclusive_start_current_segment();
        }
    }

    /// May only be called once for each time start_session is called.
    pub fn stop_session(&self) {
        let mut stopped = false;
        self.live_sessions
            .fetch_update(Ordering::Release, Ordering::Acquire, |live_sessions| {
                // We stop the current segment if we are the last sessions, but we may have to restart
                // it if in the meantime someone incremented the live_sessions counter.
                if live_sessions == 1 {
                    if !stopped {
                        self.exclusive_stop_current_segment();
                        stopped = true;
                    }
                } else {
                    if stopped {
                        self.exclusive_start_current_segment();
                        stopped = false;
                    }
                }

                Some(live_sessions - 1)
            })
            .ok();
    }
}

fn main() {
    use std::time::Duration;

    let timer = Arc::new(LiveTimer::new());

    let h1 = {
        let timer = timer.clone();
        std::thread::spawn(move || {
            timer.start_session();
            timer.stop_session();
            std::thread::sleep(Duration::from_secs(1));
            timer.start_session();
            std::thread::sleep(Duration::from_secs(1));
            timer.start_session();
            std::thread::sleep(Duration::from_secs(1));
            timer.stop_session();
            std::thread::sleep(Duration::from_secs(1));
            timer.stop_session();
            std::thread::sleep(Duration::from_secs(1));
        })
    };

    let mut recorded = [0u64; 16];

    'outer: while !h1.is_finished() {
        let mut prev = timer.total_time_live_ns();
        for i in recorded.iter_mut() {
            let new = timer.total_time_live_ns();

            if new < prev {
                break 'outer;
            }

            *i = new;
            prev = new;
        }
    }

    dbg!(&recorded);

    dbg!(h1.join().unwrap());

    dbg!(timer.total_time_live_ns());
}

// fn main() {
//     let lt = LiveTimer::new();
//     lt.start_session();
//     lt.stop_session();
//     dbg!(lt.total_time_live_ns());
//     std::thread::sleep(Duration::from_secs(1));
//     dbg!(lt.total_time_live_ns());
//     lt.start_session();
//     std::thread::sleep(Duration::from_secs(1));
//     dbg!(lt.total_time_live_ns());
//     lt.start_session();
//     std::thread::sleep(Duration::from_secs(1));
//     dbg!(lt.total_time_live_ns());
//     lt.stop_session();
//     std::thread::sleep(Duration::from_secs(1));
//     dbg!(lt.total_time_live_ns());
//     lt.stop_session();
//     std::thread::sleep(Duration::from_secs(1));
//     dbg!(lt.total_time_live_ns());
// }
