use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::thread::Thread;

#[derive(Debug)]
pub struct MemoryLimiter {
    limit: usize,
    current_usage: AtomicUsize,
    waiters: Mutex<VecDeque<WaiterType>>,
}

#[derive(Debug)]
enum WaiterType {
    Task(Waker),
    Thread(Thread),
}

impl WaiterType {
    fn wake(self) {
        match self {
            WaiterType::Task(waker) => waker.wake(),
            WaiterType::Thread(thread) => thread.unpark(),
        }
    }
}

impl Clone for MemoryLimiter {
    fn clone(&self) -> Self {
        Self {
            limit: self.limit,
            current_usage: AtomicUsize::new(self.current_usage.load(Ordering::Relaxed)),
            waiters: Mutex::new(VecDeque::new()),
        }
    }
}

impl MemoryLimiter {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            current_usage: AtomicUsize::new(0),
            waiters: Mutex::new(VecDeque::new()),
        }
    }
    
    pub fn current(&self) -> usize {
        self.current_usage.load(Ordering::Acquire)
    }
    
    pub fn limit(&self) -> usize {
        self.limit
    }
    
    pub fn try_reserve(&self, bytes: usize) -> Option<MemoryToken> {
        loop {
            let current = self.current_usage.load(Ordering::Acquire);
            
            if current + bytes > self.limit {
                return None;
            }
            
            if self.current_usage
                .compare_exchange(current, current + bytes, Ordering::AcqRel, Ordering::Acquire)
                .is_ok() {
                return Some(MemoryToken::new(self, bytes));
            }
        }
    }
    
    pub fn reserve(&self, bytes: usize) -> MemoryReserveFuture {
        MemoryReserveFuture {
            limiter: Arc::new(self.clone()),
            bytes,
        }
    }
    
    pub fn reserve_sync(&self, bytes: usize) -> MemoryToken {
        use std::thread;
        
        if let Some(token) = self.try_reserve(bytes) {
            return token;
        }
        
        loop {
            {
                let mut waiters = self.waiters.lock().unwrap();
                waiters.push_back(WaiterType::Thread(thread::current()));
            }
            
            if let Some(token) = self.try_reserve(bytes) {
                return token;
            }
            
            thread::park();
            
            if let Some(token) = self.try_reserve(bytes) {
                return token;
            }
        }
    }
    
    fn release(&self, bytes: usize) {
        self.current_usage.fetch_sub(bytes, Ordering::Release);
        
        let mut waiters = self.waiters.lock().unwrap();
        if let Some(waiter) = waiters.pop_front() {
            waiter.wake();
        }
    }
}

pub struct MemoryReserveFuture {
    limiter: Arc<MemoryLimiter>,
    bytes: usize,
}

impl Future for MemoryReserveFuture {
    type Output = MemoryToken;
    
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let limiter = &*self.limiter;
        if let Some(token) = limiter.try_reserve(self.bytes) {
            return Poll::Ready(token);
        }
        
        let mut waiters = limiter.waiters.lock().unwrap();
        waiters.push_back(WaiterType::Task(cx.waker().clone()));
        
        Poll::Pending
    }
}

#[derive(Debug, Clone)]
pub struct MemoryToken {
    limiter: Arc<MemoryLimiter>,
    bytes: usize,
    active: bool,
}

impl MemoryToken {
    fn new(limiter: &MemoryLimiter, bytes: usize) -> Self {
        Self {
            limiter: Arc::new(limiter.clone()),
            bytes,
            active: true,
        }
    }
    
    pub fn size(&self) -> usize {
        self.bytes
    }
}

impl Drop for MemoryToken {
    fn drop(&mut self) {
        if self.active {
            self.limiter.release(self.bytes);
        }
    }
}