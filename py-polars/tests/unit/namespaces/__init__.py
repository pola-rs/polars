"""
Test module containing dedicated tests for all namespace methods.

Namespace methods are methods that are available on Series and Expr classes for
operations that are only available for specific data types. For example,
`Series.str.to_lowercase()`.

These methods are almost exclusively implemented as expressions, with the Series method
dispatching to this implementation through a decorator. This means we only need to test
the Series method, as this will indirectly test the Expr method as well.
"""
