async function init() {
  const worker = new Worker("worker.js", {type: 'module'});
}
init();

