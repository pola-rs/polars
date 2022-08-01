from typing import Union

from polars import internals as pli

FrameOrSeries = Union["pli.DataFrame", "pli.Series"]


class PolarsSlice:
    """
    Apply Python slice object to Polars DataFrame or Series.

    Has full support for negative indexing and/or stride.

    """

    stop: int
    start: int
    stride: int
    slice_length: int
    is_unbounded: bool
    obj: FrameOrSeries

    def __init__(self, obj: FrameOrSeries):
        self.obj = obj

    @staticmethod
    def _as_original(lazy: "pli.LazyFrame", original: FrameOrSeries) -> FrameOrSeries:
        """Return lazy variant back to its original type."""
        frame = lazy.collect()
        return frame if isinstance(original, pli.DataFrame) else frame.to_series()

    @staticmethod
    def _lazify(obj: FrameOrSeries) -> "pli.LazyFrame":
        """Make lazy to ensure efficent/consistent handling."""
        return obj.lazy() if isinstance(obj, pli.DataFrame) else obj.to_frame().lazy()

    def _slice_positive(self, obj: "pli.LazyFrame") -> "pli.LazyFrame":
        """Logic for slices with positive stride."""
        # note: at this point stride is guaranteed to be > 1
        return obj.slice(self.start, self.slice_length).take_every(self.stride)

    def _slice_negative(self, obj: "pli.LazyFrame") -> "pli.LazyFrame":
        """Logic for slices with negative stride."""
        stride = abs(self.stride)
        lazyslice = obj.slice(self.stop + 1, self.slice_length).reverse()
        return lazyslice.take_every(stride) if (stride > 1) else lazyslice

    def _slice_setup(self, s: slice) -> None:
        """Normalise slice bounds, identify unbounded and/or zero-length slices."""
        # can normalise slice indices as we know object size
        obj_len = len(self.obj)
        start, stop, stride = slice(s.start, s.stop, s.step).indices(obj_len)

        # check if slice is actually unbounded
        if stride >= 1:
            self.is_unbounded = (start <= 0) and (stop >= obj_len)
        else:
            self.is_unbounded = (stop == -1) and (start >= obj_len - 1)

        # determine slice length
        if self.obj.is_empty():
            self.slice_length = 0
        elif self.is_unbounded:
            self.slice_length = obj_len
        else:
            self.slice_length = (
                0
                if (
                    (start == stop)
                    or (stride > 0 and start > stop)
                    or (stride < 0 and start < stop)
                )
                else abs(stop - start)
            )
        self.start, self.stop, self.stride = start, stop, stride

    def apply(self, s: slice) -> FrameOrSeries:
        """Apply a slice operation, taking advantage of any potential fast paths."""
        # normalise slice
        self._slice_setup(s)

        # check for fast-paths / single-operation calls
        if self.slice_length == 0:
            return self.obj.cleared()

        elif self.is_unbounded and self.stride in (-1, 1):
            return self.obj.reverse() if (self.stride < 0) else self.obj.clone()

        elif self.start >= 0 and self.stop >= 0 and self.stride == 1:
            return self.obj.slice(self.start, self.slice_length)

        elif self.stride < 0 and self.slice_length == 1:
            return self.obj.slice(self.stop + 1, 1)
        else:
            # multi-operation calls; make lazy
            lazyobj = self._lazify(self.obj)
            sliced = (
                self._slice_positive(lazyobj)
                if self.stride > 0
                else self._slice_negative(lazyobj)
            )
            return self._as_original(sliced, self.obj)


class LazyPolarsSlice:
    """
    Apply python slice object to Polars LazyFrame.

    Only slices with efficient computation paths mapping directly to existing lazy
    methods are supported.

    """

    obj: "pli.LazyFrame"

    def __init__(self, obj: "pli.LazyFrame"):
        self.obj = obj

    def apply(self, s: slice) -> "pli.LazyFrame":
        """
        Apply a slice operation.

        Note that LazyFrame is designed primarily for efficient computation and does not
        know its own length so, unlike DataFrame, certain slice patterns (such as those
        requiring negative stop/step) may not be supported.

        """
        start = s.start or 0
        step = s.step or 1

        # fail on operations that require length to do efficiently
        if s.stop and s.stop < 0:
            raise ValueError("Negative stop is not supported for lazy slices")
        if step < 0 and (start > 0 or s.stop is not None) and (start != s.stop):
            if not (start > 0 > step and s.stop is None):
                raise ValueError(
                    "Negative stride is not supported in conjunction with start+stop"
                )

        # ---------------------------------------
        # empty slice patterns.
        # ---------------------------------------
        # [:0]
        # [i:<=i]
        # [i:>=i:-k]
        if (step > 0 and (s.stop is not None and start >= s.stop)) or (
            step < 0 and (s.stop is not None and s.stop >= s.start >= 0)
        ):
            return self.obj.cleared()

        # ---------------------------------------
        # straight-though mappings for "reverse"
        # and/or "take_every"
        # ---------------------------------------
        # [:]    => clone()
        # [::k]  => take_every(k),
        # [::-1] => reverse(),
        # [::-k] => reverse().take_every(abs(k))
        elif start == 0 and s.stop is None:
            if step == 1:
                return self.obj.clone()
            elif step > 1:
                return self.obj.take_every(step)
            elif step == -1:
                return self.obj.reverse()
            elif step < -1:
                return self.obj.reverse().take_every(abs(step))

        elif start > 0 > step and s.stop is None:
            obj = self.obj.head(s.start + 1).reverse()
            return obj if (abs(step) == 1) else obj.take_every(abs(step))

        # ---------------------------------------
        # straight-though mappings for "head"
        # ---------------------------------------
        # [:j]    => head(j)
        # [:j:k]  => head(j).take_every(k)
        elif start == 0 and (s.stop or 0) >= 1:
            obj = self.obj.head(s.stop)
            return obj if (step == 1) else obj.take_every(step)

        # ---------------------------------------
        # straight-though mappings for "tail"
        # ---------------------------------------
        # [-i:]    => tail(abs(i))
        # [-i::k]  => tail(abs(i)).take_every(k)
        elif start < 0 and s.stop is None:
            obj = self.obj.tail(abs(start))
            return obj if (step == 1) else obj.take_every(step)

        # ---------------------------------------
        # straight-though mappings for "slice"
        # ---------------------------------------
        # [i:]     => slice(i)
        # [i:j]    => slice(i,j-i)
        # [i:j:k]  => slice(i,j-i).take_every(k)
        elif start > 0 and (s.stop is None or s.stop >= 0):
            slice_length = None if (s.stop is None) else (s.stop - start)
            obj = self.obj.slice(start, slice_length)
            return obj if (step == 1) else obj.take_every(step)

        raise ValueError(
            f"The given slice {s} is not supported by lazy computation; consider a "
            "more efficient approach, or construct explicitly with other methods"
        )
