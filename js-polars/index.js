let w = new Worker(new URL('./worker.js', import.meta.url), {
  type: 'module'
})

w.postMessage("hello")
