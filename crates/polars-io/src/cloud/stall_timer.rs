//! Stall timer for cloud reads.
//!
//! A ranged GET or a HEAD that receives no response headers, or no new body data, for longer
//! than T is abandoned: the attempt ends in a `Timeout` error and its connection is dropped.
//! object_store then retries the request, or resumes the body from the bytes already received.
//!
//! By default T sits just after the sender's second retransmission timeout:
//! 1.1 x 3 x (200 ms + `ttfb_min`), where `ttfb_min` is the smallest recent time to headers of
//! successful ranged GETs on the same object store (about 0.7 s in-region). A connection that is
//! still silent at that point is typically waiting for a lost packet to be resent, which can
//! take seconds.
//!
//! At most 20 attempts per second are abandoned per process; a stalled attempt without a token
//! keeps waiting. `POLARS_HTTP_STALL_TIMEOUT_MS=0` disables the timer, any other value sets T in
//! milliseconds. With `POLARS_LOG_HTTP_RATE_LIMIT`, the read controller's log line reports the
//! abandons per window.

use std::fmt;
use std::pin::Pin;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};
use object_store::client::{HttpError, HttpErrorKind, HttpRequest, HttpResponse, HttpResponseBody};

// The sender resends a lost packet after its retransmission timeout, which starts at about
// 200 ms (Linux' minimum) plus the round trip and doubles on every retry. The resends form a
// ladder of rungs at 1, 3, 7 and 15 times that base: 0.2, 0.6, 1.4 and 3.0 s on a short path,
// where the silences of slow S3 reads cluster. T waits for the second rung, with a margin.
const RUNG_BASE: Duration = Duration::from_millis(200);
/// Cumulative multiple of the base at the second rung (1 + 2).
const RUNGS: f64 = 3.0;
const MARGIN: f64 = 1.1;
/// `ttfb_min` before the first sample.
const TTFB_DEFAULT: Duration = Duration::from_millis(100);
const TTFB_BUCKET_US: u64 = 5_000_000;
/// Per process, with a burst of one second's worth.
const MAX_ABANDONS_PER_S: u64 = 20;

const ENV_VAR: &str = "POLARS_HTTP_STALL_TIMEOUT_MS";

enum Mode {
    Off,
    Auto,
    Fixed(Duration),
}

static MODE: LazyLock<Mode> = LazyLock::new(|| match std::env::var(ENV_VAR) {
    Err(_) => Mode::Auto,
    Ok(v) => match v.parse::<u64>() {
        Ok(0) => Mode::Off,
        Ok(ms) => Mode::Fixed(Duration::from_millis(ms)),
        Err(_) => panic!("invalid value for {ENV_VAR}: {v}"),
    },
});

/// Stall timer counts of one object store, for `POLARS_LOG_HTTP_RATE_LIMIT`.
#[derive(Debug, Default)]
pub(crate) struct StallStats {
    /// Attempts abandoned while waiting for response headers.
    hdr_abandons: AtomicU64,
    /// Attempts abandoned while waiting for body data.
    body_abandons: AtomicU64,
    /// Expired timers that got no token (`MAX_ABANDONS_PER_S`): the attempt kept waiting.
    refused: AtomicU64,
    /// T of the latest timed attempt, in µs.
    t_us: AtomicU64,
}

/// `StallStats` since the previous `take`.
#[derive(Debug)]
pub(crate) struct StallCounts {
    pub hdr_abandons: u64,
    pub body_abandons: u64,
    pub refused: u64,
    /// T of the latest timed attempt.
    pub t: Duration,
}

impl StallStats {
    pub(crate) fn take(&self) -> StallCounts {
        StallCounts {
            hdr_abandons: self.hdr_abandons.swap(0, Relaxed),
            body_abandons: self.body_abandons.swap(0, Relaxed),
            refused: self.refused.swap(0, Relaxed),
            t: Duration::from_micros(self.t_us.load(Relaxed)),
        }
    }
}

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

fn now_us() -> u64 {
    EPOCH.elapsed().as_micros() as u64
}

/// The process-wide abandon budget, as the time (µs since `EPOCH`) at which the bucket is
/// full again.
static ABANDON_BUDGET: AtomicU64 = AtomicU64::new(0);

/// Takes a token from a bucket that refills at `MAX_ABANDONS_PER_S` and holds one second's
/// worth, kept in one atomic: `full_at`, when the bucket is full again. A token moves
/// `full_at` one token's worth of time later, and is available while `full_at` stays within
/// one second of `now`.
fn take_token(full_at: &AtomicU64, now: u64) -> bool {
    const COST_US: u64 = 1_000_000 / MAX_ABANDONS_PER_S;
    let mut cur = full_at.load(Relaxed);
    loop {
        let next = cur.max(now) + COST_US;
        if next > now + 1_000_000 {
            return false;
        }
        match full_at.compare_exchange_weak(cur, next, Relaxed, Relaxed) {
            Ok(_) => return true,
            Err(v) => cur = v,
        }
    }
}

/// Takes an abandon token; without one, counts the refusal.
fn take_abandon_token(stats: &StallStats) -> bool {
    let ok = take_token(&ABANDON_BUDGET, now_us());
    if !ok {
        stats.refused.fetch_add(1, Relaxed);
    }
    ok
}

fn auto_timeout(ttfb_min: Duration) -> Duration {
    (RUNG_BASE + ttfb_min).mul_f64(MARGIN * RUNGS)
}

/// Smallest time to headers of the successful ranged GETs of one HTTP client over the last
/// 5-10 s. Time is cut into buckets of `TTFB_BUCKET_US` (numbered from 1 since `EPOCH`); the
/// current and the previous bucket each have a slot, packed as `number << 32 | minimum in µs`
/// (0: empty).
#[derive(Debug, Default)]
pub(crate) struct TtfbMin([AtomicU64; 2]);

impl TtfbMin {
    fn bucket(now: u64) -> u64 {
        now / TTFB_BUCKET_US + 1
    }

    fn add(&self, sample: Duration, now: u64) {
        let k = Self::bucket(now);
        let ours = (k << 32) | sample.as_micros().min(u32::MAX as u128) as u64;
        let slot = &self.0[(k % 2) as usize];
        let mut cur = slot.load(Relaxed);
        loop {
            // Same bucket: the smaller value is the smaller minimum. Else the slot is reused.
            let next = if cur >> 32 == k { cur.min(ours) } else { ours };
            if next == cur {
                return;
            }
            match slot.compare_exchange_weak(cur, next, Relaxed, Relaxed) {
                Ok(_) => return,
                Err(v) => cur = v,
            }
        }
    }

    fn min(&self, now: u64) -> Option<Duration> {
        let k = Self::bucket(now);
        let recent = |v: &u64| *v != 0 && (*v >> 32 == k || (*v >> 32) + 1 == k);
        let min_us = self.0.iter().map(|s| s.load(Relaxed)).filter(recent);
        min_us
            .map(|v| v & u64::from(u32::MAX))
            .min()
            .map(Duration::from_micros)
    }
}

#[derive(Debug)]
struct Stalled {
    body: bool,
    waited: Duration,
}

impl fmt::Display for Stalled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = if self.body {
            "body data"
        } else {
            "response headers"
        };
        let ms = self.waited.as_millis();
        write!(f, "stall timer: no {what} for {ms} ms ({ENV_VAR})")
    }
}

impl std::error::Error for Stalled {}

/// Counts the abandon and returns the error that ends the attempt.
fn abandon(stats: &StallStats, body: bool, waited: Duration) -> HttpError {
    let count = if body {
        &stats.body_abandons
    } else {
        &stats.hdr_abandons
    };
    count.fetch_add(1, Relaxed);
    HttpError::new(HttpErrorKind::Timeout, Stalled { body, waited })
}

/// The timer of one HTTP attempt: it times the wait for the response headers, then hands the
/// body to a `StallBody`.
pub(crate) struct StallTimer<'a> {
    t: Duration,
    /// Ranged GET, else HEAD.
    get: bool,
    ttfb_min: &'a TtfbMin,
    stats: &'a Arc<StallStats>,
}

impl<'a> StallTimer<'a> {
    /// `None` if off, or unless `req` is a ranged GET or a HEAD.
    pub(crate) fn start(
        req: &HttpRequest,
        ttfb_min: &'a TtfbMin,
        stats: &'a Arc<StallStats>,
    ) -> Option<Self> {
        let t = match *MODE {
            Mode::Off => return None,
            Mode::Auto => None,
            Mode::Fixed(t) => Some(t),
        };
        let get = match req.method().as_str() {
            "GET" if req.headers().contains_key("range") => true,
            "HEAD" => false,
            _ => return None,
        };
        let t = t.unwrap_or_else(|| auto_timeout(ttfb_min.min(now_us()).unwrap_or(TTFB_DEFAULT)));
        let t_us = t.as_micros() as u64;
        if stats.t_us.load(Relaxed) != t_us {
            stats.t_us.store(t_us, Relaxed);
        }
        Some(Self {
            t,
            get,
            ttfb_min,
            stats,
        })
    }

    /// Awaits `fut`, the request up to its response headers. At T without headers the attempt
    /// is abandoned if there is a token, else it waits another T. A successful ranged GET adds
    /// its time to headers to `ttfb_min`.
    pub(crate) async fn headers(
        &self,
        fut: impl Future<Output = Result<HttpResponse, HttpError>>,
    ) -> Result<HttpResponse, HttpError> {
        let start = Instant::now();
        let mut fut = std::pin::pin!(fut);
        let response = loop {
            match tokio::time::timeout(self.t, fut.as_mut()).await {
                Ok(response) => break response,
                Err(_) if take_abandon_token(self.stats) => {
                    return Err(abandon(self.stats, false, start.elapsed()));
                },
                Err(_) => {},
            }
        };
        if self.get && response.as_ref().is_ok_and(|r| r.status().is_success()) {
            self.ttfb_min.add(start.elapsed(), now_us());
        }
        response
    }

    /// The body of a successful ranged GET goes into a `StallBody`; other responses pass
    /// through.
    pub(crate) fn body(
        self,
        response: Result<HttpResponse, HttpError>,
    ) -> Result<HttpResponse, HttpError> {
        match response {
            Ok(r) if self.get && r.status().is_success() => {
                let (parts, inner) = r.into_parts();
                let body = StallBody::new(self.t, Arc::clone(self.stats), inner);
                Ok(HttpResponse::from_parts(parts, HttpResponseBody::new(body)))
            },
            response => response,
        }
    }
}

/// A response body that ends in a `Timeout` error when no data frame arrives for T while the
/// reader waits. The clock runs from the first `Pending` after a data frame (or of the body)
/// to the next data frame; a reader that does not poll is not timed. One `Sleep` per body,
/// moved only when it fires, so a body that keeps flowing costs no timer operations.
struct StallBody {
    t: Duration,
    stats: Arc<StallStats>,
    inner: HttpResponseBody,
    sleep: Option<Pin<Box<tokio::time::Sleep>>>,
    /// (start, due) of the current wait.
    wait: Option<(Instant, Instant)>,
}

impl StallBody {
    fn new(t: Duration, stats: Arc<StallStats>, inner: HttpResponseBody) -> Self {
        Self {
            t,
            stats,
            inner,
            sleep: None,
            wait: None,
        }
    }
}

impl Body for StallBody {
    type Data = Bytes;
    type Error = HttpError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, HttpError>>> {
        let this = &mut *self;
        let out = Pin::new(&mut this.inner).poll_frame(cx);
        match &out {
            Poll::Ready(Some(Ok(frame))) if frame.is_data() => this.wait = None,
            Poll::Pending => {
                let t = this.t;
                let (start, mut due) = *this.wait.get_or_insert_with(|| {
                    let now = Instant::now();
                    (now, now + t)
                });
                let sleep = this
                    .sleep
                    .get_or_insert_with(|| Box::pin(tokio::time::sleep_until(due.into())));
                // The sleep may still point at an earlier wait's due time: then it is moved to
                // this wait's. Once due without a token, the wait gets another T.
                while sleep.as_mut().poll(cx).is_ready() {
                    let now = Instant::now();
                    if now >= due {
                        if take_abandon_token(&this.stats) {
                            let e = abandon(&this.stats, true, now - start);
                            return Poll::Ready(Some(Err(e)));
                        }
                        due = now + t;
                        this.wait = Some((start, due));
                    }
                    sleep.as_mut().reset(due.into());
                }
            },
            _ => {},
        }
        out
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures::stream::{BoxStream, StreamExt};

    use super::*;

    struct StreamBody(Mutex<BoxStream<'static, Bytes>>);

    impl Body for StreamBody {
        type Data = Bytes;
        type Error = HttpError;

        fn poll_frame(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, HttpError>>> {
            let chunks = self.get_mut().0.get_mut().unwrap();
            chunks
                .poll_next_unpin(cx)
                .map(|x| x.map(|b| Ok(Frame::data(b))))
        }
    }

    fn timed(
        chunks: BoxStream<'static, Bytes>,
        t: Duration,
        stats: &Arc<StallStats>,
    ) -> HttpResponseBody {
        let inner = HttpResponseBody::new(StreamBody(Mutex::new(chunks)));
        HttpResponseBody::new(StallBody::new(t, Arc::clone(stats), inner))
    }

    #[tokio::test]
    async fn stall_timer_body() {
        let ms = Duration::from_millis;
        let stats = Arc::new(StallStats::default());
        let stalled =
            futures::stream::iter([Bytes::from_static(b"a")]).chain(futures::stream::pending());
        let err = timed(stalled.boxed(), ms(50), &stats)
            .bytes()
            .await
            .unwrap_err();
        assert_eq!(err.kind(), HttpErrorKind::Timeout);
        let c = stats.take();
        assert_eq!((c.hdr_abandons, c.body_abandons, c.refused), (0, 1, 0));

        let slow = futures::stream::iter(0..5).then(move |_| async move {
            tokio::time::sleep(ms(20)).await;
            Bytes::from_static(b"a")
        });
        let bytes = timed(slow.boxed(), ms(200), &stats).bytes().await.unwrap();
        assert_eq!(bytes.len(), 5);
        let c = stats.take();
        assert_eq!((c.hdr_abandons, c.body_abandons, c.refused), (0, 0, 0));
    }

    #[test]
    fn stall_timer_budget_and_ttfb_min() {
        let s = 1_000_000;
        let budget = AtomicU64::new(0);
        let taken = |now| (0..30).filter(|_| take_token(&budget, now)).count();
        assert_eq!(taken(10 * s), 20);
        assert_eq!(taken(10 * s + s / 2), 10);

        let ms = Duration::from_millis;
        let t = TtfbMin::default();
        assert_eq!(t.min(0), None);
        t.add(ms(30), 0);
        t.add(ms(20), s);
        t.add(ms(25), 6 * s);
        assert_eq!(t.min(6 * s), Some(ms(20)));
        assert_eq!(t.min(11 * s), Some(ms(25)));
        assert_eq!(t.min(16 * s), None);
    }
}
