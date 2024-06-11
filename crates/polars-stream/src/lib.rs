#[allow(unused)]
mod executor;
#[allow(unused)]
mod async_primitives;


pub async fn dummy() {
    let num_threads = 8;
    executor::set_num_threads(num_threads);
    executor::task_scope(|s| {
        s.spawn_task(false, async { });
    });
    
    let (mut send, mut recv) = async_primitives::pipe::pipe::<u32>();
    send.send(42).await.ok();
    recv.recv().await.ok();
    let (mut send, mut recvs) = async_primitives::distributor_channel::distributor_channel::<u32>(num_threads, 8);
    send.send(42).await.ok();
    recvs[0].recv().await.ok();
}