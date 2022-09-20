import abc
from collections import deque
from collections.abc import Iterator
from itertools import chain, islice


class RollingObject(Iterator):
    """
    Baseclass for rolling iterator objects.
    All of the iteration logic specific to 'fixed' and
    'variable' window types is handled by this class.
    Subclasses of RollingObject must implement methods
    to initialize and manipulate any attributes needed
    to compute the window value as it rolls across the
    given iterable.
    These methods are:
     * _init_fixed()     fixed window initialization
     * _init_variable()  variable window initialization
     * _update_window()  add new value, remove oldest value
     * _add_new()        add new value (increase size)
     * _remove_old()     remove oldest value (decrease size)
    The following @property methods must also be implemented:
     * _obs              number of observations in window
     * current_value     current value of operation on window
    """

    def __init__(self, iterable, window_size, window_type="fixed", **kwargs):
        self.window_type = window_type
        self.window_size = _validate_window_size(window_size)
        self._iterator = iter(iterable)
        self._filled = self.window_type == "fixed"

        if window_type == "fixed":
            self._init_fixed(iterable, window_size, **kwargs)

        elif window_type == "variable":
            self._init_variable(iterable, window_size, **kwargs)

        else:
            raise ValueError(f"Unknown window_type '{window_type}'")

    def __repr__(self):
        return "Rolling(operation='{}', window_size={}, window_type='{}')".format(
            self.__class__.__name__, self.window_size, self.window_type
        )

    def _next_fixed(self):
        new = next(self._iterator)
        self._update_window(new)
        return self.current_value

    def _next_variable(self):
        # while the window size is not reached, add new values
        if not self._filled and self._obs < self.window_size:
            new = next(self._iterator)
            self._add_new(new)
            self._filled = self._obs == self.window_size
            return self.current_value

        # once the window size is reached, consider fixed until iterator ends
        try:
            return self._next_fixed()

        # if the iterator finishes, remove the oldest values one at a time
        except StopIteration:
            if self._obs == 1:
                raise
            else:
                self._remove_old()
                return self.current_value

    def __next__(self):

        if self.window_type == "fixed":
            return self._next_fixed()

        if self.window_type == "variable":
            return self._next_variable()

        raise NotImplementedError(f"next() not implemented for {self.window_type}")

    def extend(self, iterable):
        """
        Extend the iterator being consumed with a new iterable.
        The extend() method may be called at any time (even after
        StopIteration has been raised). The most recent values from
        the current iterator are retained and used in the calculation
        of the next window value.
        For "variable" windows which are decreasing in size, extending
        the iterator means that these windows will grow towards their
        maximum size again.
        """
        self._iterator = chain(self._iterator, iterable)

        if self.window_type == "variable":
            self._filled = False

    @property
    @abc.abstractmethod
    def current_value(self):
        """
        Return the current value of the window
        """
        pass

    @property
    @abc.abstractmethod
    def _obs(self):
        """
        Return the number of observations in the window
        """
        pass

    @abc.abstractmethod
    def _init_fixed(self):
        """
        Intialise as a fixed-size window
        """
        pass

    @abc.abstractmethod
    def _init_variable(self):
        """
        Intialise as a variable-size window
        """
        pass

    @abc.abstractmethod
    def _remove_old(self):
        """
        Remove the oldest value from the window, decreasing window size by 1
        """
        pass

    @abc.abstractmethod
    def _add_new(self, new):
        """
        Add a new value to the window, increasing window size by 1
        """
        pass

    @abc.abstractmethod
    def _update_window(self, new):
        """
        Add a new value to the window and remove the oldest value from the window
        """
        pass


def _validate_window_size(k):
    """
    Check if k is a positive integer
    """
    if not isinstance(k, int):
        raise TypeError(f"window_size must be integer type, got {type(k).__name__}")
    if k <= 0:
        raise ValueError("window_size must be positive")
    return k


class Kurtosis(RollingObject):
    """Iterator object that computes the kurtosis
    of a rolling window over a Python iterable.
    Parameters
    ----------
    iterable : any iterable object
    window_size : integer, the size of the rolling
        window moving over the iterable (must be
        greater than 3)
    Complexity
    ----------
    Update time:  O(1)
    Memory usage: O(k)
    where k is the size of the rolling window
    Notes
    -----
    This implementation of rolling skewness is based
    on the pandas code here:
    https://github.com/pandas-dev/pandas/blob/master/pandas/_libs/window.pyx
    """

    def _init_fixed(self, iterable, window_size, **kwargs):
        if window_size <= 3:
            raise ValueError("window_size must be greater than 3")

        self._buffer = deque(maxlen=window_size)
        self._x1 = 0.0
        self._x2 = 0.0
        self._x3 = 0.0
        self._x4 = 0.0

        for new in islice(self._iterator, window_size - 1):
            self._add_new(new)

        # insert zero at the start of the buffer so that the
        # the first call to update returns the correct value
        self._buffer.appendleft(0)

    def _init_variable(self, iterable, window_size, **kwargs):
        if window_size <= 3:
            raise ValueError("window_size must be greater than 3")

        self._buffer = deque(maxlen=window_size)
        self._x1 = 0.0
        self._x2 = 0.0
        self._x3 = 0.0
        self._x4 = 0.0

    def _add_new(self, new):
        self._buffer.append(new)

        self._x1 += new
        self._x2 += new * new
        self._x3 += new**3
        self._x4 += new**4

    def _remove_old(self):
        old = self._buffer.popleft()

        self._x1 -= old
        self._x2 -= old * old
        self._x3 -= old**3
        self._x4 -= old**4

    def _update_window(self, new):
        old = self._buffer[0]
        self._buffer.append(new)

        self._x1 += new - old
        self._x2 += new * new - old * old
        self._x3 += new**3 - old**3
        self._x4 += new**4 - old**4

    @property
    def current_value(self):
        N = self._obs

        if N <= 3:
            return float("nan")

        # compute moments
        A = self._x1 / N
        R = A * A

        B = self._x2 / N - R
        R *= A

        C = self._x3 / N - R - 3 * A * B

        R *= A

        D = self._x4 / N - R - 6 * B * A * A - 4 * C * A

        if B <= 1e-14:
            return float("nan")

        K = (N * N - 1) * D / (B * B) - 3 * ((N - 1) ** 2)

        print(K)
        # print(N, self._x3, A, B, C, D, K)

        K = K / ((N - 2) * (N - 3))

        print(K, ((N - 2) * (N - 3)))

        return K

    @property
    def _obs(self):
        return len(self._buffer)


k = Kurtosis([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], 4)
print(list(k))
