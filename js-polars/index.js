async function init() {
  const worker = new Worker("worker.js", {type: 'module'});
  worker.addEventListener('message', async ({data}) => {
    console.log(data)
  })
}
init();

