# Multiprocessing

TLDR: if you find that using Python's built-in `multiprocessing` module together with Polars results in a Polars error about multiprocessing methods, you should make sure you are using `spawn`, not `fork`, as the starting method:

{{code_block('user-guide/misc/multiprocess','recommendation',[])}}

## When not to use multiprocessing

Before we dive into the details, it is important to emphasize that Polars has been built from the start to use all your CPU cores.
It does this by executing computations which can be done in parallel in separate threads.
For example, requesting two expressions in a `select` statement can be done in parallel, with the results only being combined at the end.
Another example is aggregating a value within groups using `group_by().agg(<expr>)`, each group can be evaluated separately.
It is very unlikely that the `multiprocessing` module can improve your code performance in these cases.

See [the optimizations section](../lazy/optimizations.md) for more optimizations.

## When to use multiprocessing

Although Polars is multithreaded, other libraries may be single-threaded.
When the other library is the bottleneck, and the problem at hand is parallelizable, it makes sense to use multiprocessing to gain a speed up.

## The problem with the default multiprocessing config

### Summary

The [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html) lists the three methods to create a process pool:

1. spawn
1. fork
1. forkserver

The description of fork is (as of 2022-10-15):

> The parent process uses os.fork() to fork the Python interpreter. The child process, when it begins, is effectively identical to the parent process. All resources of the parent are inherited by the child process. Note that safely forking a multithreaded process is problematic.

> Available on Unix only. The default on Unix.

The short summary is: Polars is multithreaded as to provide strong performance out-of-the-box.
Thus, it cannot be combined with `fork`.
If you are on Unix (Linux, BSD, etc), you are using `fork`, unless you explicitly override it.

The reason you may not have encountered this before is that pure Python code, and most Python libraries, are (mostly) single threaded.
Alternatively, you are on Windows or MacOS, on which `fork` is not even available as a method (for MacOS it was up to Python 3.7).

Thus one should use `spawn`, or `forkserver`, instead. `spawn` is available on all platforms and the safest choice, and hence the recommended method.

### Example

The problem with `fork` is in the copying of the parent's process.
Consider the example below, which is a slightly modified example posted on the [Polars issue tracker](https://github.com/pola-rs/polars/issues/3144):

{{code_block('user-guide/misc/multiprocess','example1',[])}}

Using `fork` as the method, instead of `spawn`, will cause a dead lock.
Please note: Polars will not even start and raise the error on multiprocessing method being set wrong, but if the check had not been there, the deadlock would exist.

The fork method is equivalent to calling `os.fork()`, which is a system call as defined in [the POSIX standard](https://pubs.opengroup.org/onlinepubs/9699919799/functions/fork.html):

> A process shall be created with a single thread. If a multi-threaded process calls fork(), the new process shall contain a replica of the calling thread and its entire address space, possibly including the states of mutexes and other resources. Consequently, to avoid errors, the child process may only execute async-signal-safe operations until such time as one of the exec functions is called.

In contrast, `spawn` will create a completely new fresh Python interpreter, and not inherit the state of mutexes.

So what happens in the code example?
For reading the file with `pl.read_parquet` the file has to be locked.
Then `os.fork()` is called, copying the state of the parent process, including mutexes.
Thus all child processes will copy the file lock in an acquired state, leaving them hanging indefinitely waiting for the file lock to be released, which never happens.

What makes debugging these issues tricky is that `fork` can work.
Change the example to not having the call to `pl.read_parquet`:

{{code_block('user-guide/misc/multiprocess','example2',[])}}

This works fine.
Therefore debugging these issues in larger code bases, i.e. not the small toy examples here, can be a real pain, as a seemingly unrelated change can break your multiprocessing code.
In general, one should therefore never use the `fork` start method with multithreaded libraries unless there are very specific requirements that cannot be met otherwise.

### Pro's and cons of fork

Based on the example, you may think, why is `fork` available in Python to start with?

First, probably because of historical reasons: `spawn` was added to Python in version 3.4, whilst `fork` has been part of Python from the 2.x series.

Second, there are several limitations for `spawn` and `forkserver` that do not apply to `fork`, in particular all arguments should be pickable.
See the [Python multiprocessing docs](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods) for more information.

Third, because it is faster to create new processes compared to `spawn`, as `spawn` is effectively `fork` + creating a brand new Python process without the locks by calling [execv](https://pubs.opengroup.org/onlinepubs/9699919799/functions/exec.html).
Hence the warning in the Python docs that it is slower: there is more overhead to `spawn`.
However, in almost all cases, one would like to use multiple processes to speed up computations that take multiple minutes or even hours, meaning the overhead is negligible in the grand scheme of things.
And more importantly, it actually works in combination with multithreaded libraries.

Fourth, `spawn` starts a new process, and therefore it requires code to be importable, in contrast to `fork`.
In particular, this means that when using `spawn` the relevant code should not be in the global scope, such as in Jupyter notebooks or in plain scripts.
Hence in the examples above, we define functions where we spawn within, and run those functions from a `__main__` clause.
This is not an issue for typical projects, but during quick experimentation in notebooks it could fail.

## References

1. https://docs.python.org/3/library/multiprocessing.html

1. https://pythonspeed.com/articles/python-multiprocessing/

1. https://pubs.opengroup.org/onlinepubs/9699919799/functions/fork.html

1. https://bnikolic.co.uk/blog/python/parallelism/2019/11/13/python-forkserver-preload.html
