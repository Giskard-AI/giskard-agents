"""Composable rate limiters and scoped registry.

This module provides async rate limiter primitives that can be combined and
registered by target path to enforce both global and scoped limits.
"""

import asyncio
import time
from collections import defaultdict
from contextlib import contextmanager

from pydantic import BaseModel


class RateLimitEntry(BaseModel, frozen=True):
    """Single rate limit contribution.

    Parameters
    ----------
    name : str
        Human-readable limiter identifier.
    wait_time : float
        Time waited (in seconds) due to this limiter.
    """

    name: str
    wait_time: float


class RateLimitDetails(BaseModel, frozen=True):
    """Aggregated rate limit details for a request.

    Parameters
    ----------
    entries : list[RateLimitEntry]
        List of limiter contributions.
    """

    entries: list[RateLimitEntry] = []

    @property
    def wait_time(self) -> float:
        """Total wait time.

        Returns
        -------
        float
            Sum of wait times for all entries.
        """

        return sum(e.wait_time for e in self.entries)

    def __add__(self, other: "RateLimitDetails") -> "RateLimitDetails":
        """Combine two rate limit details.

        Parameters
        ----------
        other : RateLimitDetails
            Details to merge.

        Returns
        -------
        RateLimitDetails
            Combined details with concatenated entries.
        """

        return RateLimitDetails(entries=self.entries + other.entries)


NO_WAIT_TIME = RateLimitDetails(entries=[])


class RateLimiter:
    """Base class for async rate limiters."""

    async def acquire(self) -> RateLimitDetails:
        """Acquire rate limiter capacity.

        Returns
        -------
        RateLimitDetails
            Details about any wait time incurred.
        """

        return NO_WAIT_TIME

    async def release(self) -> None:
        """Release any resources associated with a limiter."""

    async def __aenter__(self) -> RateLimitDetails:
        """Enter the async context, acquiring rate limits.

        Returns
        -------
        RateLimitDetails
            Details about any wait time incurred.
        """

        return await self.acquire()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the async context, releasing any resources."""

        await self.release()

    @classmethod
    def budget(cls, rpm: int, max_concurrent: int | None = None) -> "RateLimiter":
        """Create a composed limiter for both RPM and concurrency.

        Parameters
        ----------
        rpm : int
            Requests per minute.
        max_concurrent : int | None
            Maximum concurrent requests, if provided.

        Returns
        -------
        RateLimiter
            A composite limiter.
        """

        rate_limiter = cls.rpm(rpm)

        if max_concurrent is not None:
            # Max concurrent requests should be applied first
            rate_limiter = cls.max_in_flight(max_concurrent).with_rate_limiters(
                rate_limiter
            )

        return rate_limiter

    @classmethod
    def rpm(cls, rpm: int) -> "MaxRequestsPerMinute":
        """Create a requests-per-minute limiter.

        Parameters
        ----------
        rpm : int
            Requests per minute.

        Returns
        -------
        MaxRequestsPerMinute
            RPM limiter.
        """

        return MaxRequestsPerMinute(rpm=rpm)

    @classmethod
    def max_in_flight(cls, max_concurrent: int) -> "MaxConcurrentRequests":
        """Create a maximum concurrency limiter.

        Parameters
        ----------
        max_concurrent : int
            Maximum concurrent requests.

        Returns
        -------
        MaxConcurrentRequests
            Concurrency limiter.
        """

        return MaxConcurrentRequests(max_concurrent=max_concurrent)

    def with_rate_limiters(
        self, *rate_limiters: "RateLimiter"
    ) -> "CompositeRateLimiter":
        """Compose this limiter with others.

        Parameters
        ----------
        *rate_limiters : RateLimiter
            Additional limiters to apply after this one.

        Returns
        -------
        CompositeRateLimiter
            Composite limiter.
        """

        return CompositeRateLimiter(self, *rate_limiters)


class MaxRequestsPerMinute(RateLimiter):
    """Enforce a minimum interval between request starts."""

    rpm: int
    _lock: asyncio.Lock
    _next_request_time: float

    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValueError("rpm must be greater than 0")

        self.rpm = rpm
        self._min_interval = 60.0 / self.rpm
        self._lock = asyncio.Lock()
        self._next_request_time = time.monotonic()

    async def acquire(self) -> RateLimitDetails:
        """Acquire capacity based on request spacing.

        Returns
        -------
        RateLimitDetails
            Details about any wait time incurred.
        """

        async with self._lock:
            now = time.monotonic()
            if self._next_request_time <= now:
                self._next_request_time = now + self._min_interval
                return NO_WAIT_TIME

            wait_time = self._next_request_time - now
            await asyncio.sleep(wait_time)
            self._next_request_time += self._min_interval
            return RateLimitDetails(
                entries=[
                    RateLimitEntry(
                        name=f"MaxRequestsPerMinute(rpm={self.rpm})",
                        wait_time=wait_time,
                    )
                ]
            )


class MaxConcurrentRequests(RateLimiter):
    """Enforce a maximum number of concurrent in-flight requests."""

    max_concurrent: int
    _semaphore: asyncio.Semaphore

    def __init__(self, max_concurrent: int):
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")

        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def acquire(self) -> RateLimitDetails:
        """Acquire a concurrency slot.

        Returns
        -------
        RateLimitDetails
            Details about any wait time incurred.
        """

        start_time = time.monotonic()
        await self._semaphore.acquire()
        end_time = time.monotonic()
        wait_time = end_time - start_time
        if wait_time < 1e-3:
            # If the wait time is less than 1ms, consider the request not throttled
            return NO_WAIT_TIME

        return RateLimitDetails(
            entries=[
                RateLimitEntry(
                    name=f"MaxConcurrentRequests(max_concurrent={self.max_concurrent})",
                    wait_time=wait_time,
                )
            ]
        )

    async def release(self) -> None:
        """Release a concurrency slot."""

        self._semaphore.release()


class CompositeRateLimiter(RateLimiter):
    """Apply multiple rate limiters in sequence."""

    rate_limiters: list[RateLimiter]

    def __init__(self, *rate_limiters: RateLimiter):
        self.rate_limiters = rate_limiters

    async def acquire(self) -> RateLimitDetails:
        """Acquire all composed limiters.

        Returns
        -------
        RateLimitDetails
            Combined details from all limiters.
        """

        details = NO_WAIT_TIME
        for rate_limiter in self.rate_limiters:
            details += await rate_limiter.acquire()
        return details

    async def release(self) -> None:
        """Release all composed limiters in reverse order."""

        for rate_limiter in reversed(self.rate_limiters):
            await rate_limiter.release()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeRateLimiter):
            return False
        return self.rate_limiters == other.rate_limiters

    def __hash__(self) -> int:
        return hash(tuple(self.rate_limiters))


class _RateLimiterRegistry:
    _rate_limiters: list[RateLimiter]
    _children: dict[str, "_RateLimiterRegistry"]

    def __init__(self):
        self._rate_limiters = []
        self._children = defaultdict(lambda: _RateLimiterRegistry())

    def list_by_target(self, *target: str) -> list[RateLimiter]:
        if len(target) == 0:
            return self._rate_limiters

        return self._rate_limiters + self._children[target[0]].list_by_target(
            *target[1:]
        )

    def register_rate_limiter(self, rate_limiter: RateLimiter, *target: str) -> None:
        if len(target) == 0:
            self._rate_limiters.append(rate_limiter)
            return

        self._children[target[0]].register_rate_limiter(rate_limiter, *target[1:])

    def unregister_rate_limiter(self, rate_limiter: RateLimiter, *target: str) -> bool:
        if len(target) == 0:
            if rate_limiter in self._rate_limiters:
                self._rate_limiters.remove(rate_limiter)
                return True
            return False

        return self._children[target[0]].unregister_rate_limiter(
            rate_limiter, *target[1:]
        )


_RATE_LIMITER_REGISTRY = _RateLimiterRegistry()


def throttle(*target: str) -> RateLimiter:
    """Build a composite limiter for the requested target.

    Parameters
    ----------
    *target : str
        Target path components.

    Returns
    -------
    RateLimiter
        Composite limiter for all matching scoped limiters.
    """

    return CompositeRateLimiter(*_RATE_LIMITER_REGISTRY.list_by_target(*target))


def register_rate_limiter(rate_limiter: RateLimiter, *target: str) -> None:
    """Register a limiter for a target scope.

    Parameters
    ----------
    rate_limiter : RateLimiter
        Limiter to register.
    *target : str
        Target path components.
    """

    _RATE_LIMITER_REGISTRY.register_rate_limiter(rate_limiter, *target)


def unregister_rate_limiter(rate_limiter: RateLimiter, *target: str) -> bool:
    """Unregister a limiter from a target scope.

    Parameters
    ----------
    rate_limiter : RateLimiter
        Limiter to unregister.
    *target : str
        Target path components.

    Returns
    -------
    bool
        True if a limiter was removed.
    """

    return _RATE_LIMITER_REGISTRY.unregister_rate_limiter(rate_limiter, *target)


@contextmanager
def scoped_limiter(rate_limiter: RateLimiter, *target: str):
    """Temporarily register a limiter within a scope.

    Parameters
    ----------
    rate_limiter : RateLimiter
        Limiter to register.
    *target : str
        Target path components.

    Yields
    ------
    RateLimiter
        The registered limiter.
    """

    register_rate_limiter(rate_limiter, *target)
    try:
        yield rate_limiter
    finally:
        unregister_rate_limiter(rate_limiter, *target)
