import asyncio
import sys
import time

import pytest
from giskard.agents.rate_limiter import (
    NO_WAIT_TIME,
    CompositeRateLimiter,
    RateLimiter,
    scoped_limiter,
    throttle,
)

JITTER_TIME = 0.02  # 20ms jitter


class TestMaxRequestsPerMinute:
    @pytest.mark.parametrize("rpm", [0, -1, -(sys.maxsize / 2)])
    def test_rpm_cannot_be_less_than_1(self, rpm: int):
        with pytest.raises(ValueError, match="rpm must be greater than 0"):
            RateLimiter.rpm(rpm)

    @pytest.mark.timeout(1)
    async def test_allow_parallel_requests(self):
        job_started_signal = asyncio.Event()
        signal = asyncio.Event()

        async def wait_for_signal(rate_limiter: RateLimiter) -> None:
            async with rate_limiter:
                # Ensure the job has started
                job_started_signal.set()
                await signal.wait()

        async def signal_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter:
                await asyncio.sleep(1e-3)
                signal.set()

        rate_limiter = RateLimiter.rpm(1_000_000)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(wait_for_signal(rate_limiter))
            # Ensure the job has started so it'll be a deadlock if the rate limiter is not working with parallel tasks
            await job_started_signal.wait()
            tg.create_task(signal_task(rate_limiter))

    async def test_throttle_rate(self):
        rate_limiter = RateLimiter.budget(rpm=6_000)  # 100 requests per second

        all_details = []

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter as details:
                all_details.append(details)

        start_time = time.monotonic()
        async with asyncio.TaskGroup() as tg:
            for _ in range(50):
                tg.create_task(throttle_task(rate_limiter))

        elapsed_time = time.monotonic() - start_time
        assert (
            elapsed_time >= 0.49
        )  # 49 requests should take at least 490ms, first request is free, so we need to wait for 49 requests
        assert (
            elapsed_time < 0.49 + JITTER_TIME
        )  # Ensure reasonable time to run the tasks

        assert len(all_details) == 50
        assert (
            all_details[0].wait_time == 0
        )  # Ensure the first request is not throttled
        for detail in all_details[1:]:
            assert detail.wait_time > 0
            assert (
                detail.wait_time <= 0.49
            )  # Ensure the wait time is within the expected range
            assert (
                detail.entries[0].name == "MaxRequestsPerMinute(rpm=6000)"
            )  # Ensure the rate limiter name is correct

    async def test_throttle_rate_reset_after_interval(self):
        rate_limiter = RateLimiter.rpm(60 * 25)  # 25 requests per second

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter as details:
                return details

        for _ in range(10):
            details = await throttle_task(rate_limiter)
            assert details.wait_time == 0  # Ensure the task is not throttled
            await asyncio.sleep(rate_limiter._min_interval)


class TestMaxConcurrentRequests:
    @pytest.mark.parametrize("max_concurrent", [0, -1, -(sys.maxsize / 2)])
    def test_max_concurrent_cannot_be_less_than_1(self, max_concurrent: int):
        with pytest.raises(ValueError, match="max_concurrent must be greater than 0"):
            RateLimiter.max_in_flight(max_concurrent)

    @pytest.mark.timeout(1)
    async def test_allow_parallel_requests(self):
        rate_limiter = RateLimiter.max_in_flight(10)
        barrier = asyncio.Barrier(10)

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter as details:
                assert details.wait_time == 0  # Ensure the task is not throttled
                await barrier.wait()

        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                tg.create_task(throttle_task(rate_limiter))

    @pytest.mark.timeout(1)
    async def test_block_when_max_concurrent_reached(self):
        rate_limiter = RateLimiter.max_in_flight(5)
        barrier = asyncio.Barrier(10)
        waited_tasks = []

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter as details:
                waited_tasks.append(details)
                await barrier.wait()

        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                tg.create_task(throttle_task(rate_limiter))

            await asyncio.sleep(JITTER_TIME)
            assert barrier.n_waiting == 5  # ensure that only 5 tasks are waiting
            assert len(waited_tasks) == 5  # ensure that 5 tasks have been throttled
            for detail in waited_tasks:
                assert detail.wait_time == 0

            for _ in range(5):
                tg.create_task(barrier.wait())  # Unblock the waiting tasks

            await asyncio.sleep(JITTER_TIME)
            assert barrier.n_waiting == 5  # Ensure the next 5 tasks have been started

            for _ in range(5):
                tg.create_task(barrier.wait())  # Unblock the waiting tasks

            await asyncio.sleep(JITTER_TIME)
            assert barrier.n_waiting == 0  # Ensure all tasks have been finished

            assert len(waited_tasks) == 10  # Ensure 5 tasks have been throttled
            for detail in waited_tasks[5:]:
                assert (
                    detail.wait_time > JITTER_TIME
                    and detail.wait_time <= 2 * JITTER_TIME
                )  # Ensure the wait time is within the expected range (more than jitter time, less than 2 * jitter time)
                assert (
                    detail.entries[0].name == "MaxConcurrentRequests(max_concurrent=5)"
                )  # Ensure the rate limiter name is correct


class TestCompositeRateLimiter:
    @pytest.mark.timeout(1)
    async def test_composite_rate_limiter(self):
        rate_limiter = RateLimiter.budget(rpm=6_000, max_concurrent=5)

        barrier = asyncio.Barrier(10)
        waited_tasks = []

        async def throttle_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter as details:
                waited_tasks.append(details)
                await barrier.wait()
                return details

        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                tg.create_task(throttle_task(rate_limiter))

            expected_wait_time = 0.04 + JITTER_TIME
            await asyncio.sleep(expected_wait_time)
            assert barrier.n_waiting == 5  # ensure that only 5 tasks are waiting
            assert len(waited_tasks) == 5  # ensure that 5 tasks have been throttled
            assert (
                waited_tasks[0].wait_time == 0
            )  # ensure the first task is not throttled
            for detail in waited_tasks[1:]:
                assert (
                    detail.wait_time > 0 and detail.wait_time <= 0.04
                )  # ensure the wait time is within the expected range (10ms throttle per task)

            for _ in range(5):
                tg.create_task(barrier.wait())  # Unblock the waiting tasks

            await asyncio.sleep(expected_wait_time)
            assert (
                barrier.n_waiting == 5
            )  # ensure that the next 5 tasks have been started
            assert len(waited_tasks) == 10  # Ensure that all tasks have been throttled

            # ensure the first task is not throttled
            assert len(waited_tasks[5].entries) == 1
            assert (
                waited_tasks[5].entries[0].name
                == "MaxConcurrentRequests(max_concurrent=5)"
            )  # ensure the rate limiter name is correct
            assert (
                waited_tasks[5].entries[0].wait_time > expected_wait_time
                and waited_tasks[5].entries[0].wait_time <= 2 * expected_wait_time
            )  # ensure the wait time is within the expected range (more than jitter time, less than 2 * jitter time)
            for detail in waited_tasks[6:]:
                assert len(detail.entries) == 2
                assert (
                    detail.entries[0].name == "MaxConcurrentRequests(max_concurrent=5)"
                )  # ensure the rate limiter name is correct
                assert (
                    detail.entries[0].wait_time > expected_wait_time
                    and detail.entries[0].wait_time <= 2 * expected_wait_time
                )  # ensure the wait time is within the expected range (more than jitter time, less than 2 * jitter time)
                assert (
                    detail.entries[1].name == "MaxRequestsPerMinute(rpm=6000)"
                )  # ensure the rate limiter name is correct
                assert (
                    detail.entries[1].wait_time > 0
                    and detail.entries[1].wait_time <= 0.04
                )  # ensure the wait time is within the expected range (10ms throttle per task)
                assert (
                    detail.wait_time
                    == detail.entries[0].wait_time + detail.entries[1].wait_time
                )  # ensure the wait time is the sum of the two rate limiters

            for _ in range(5):
                tg.create_task(barrier.wait())  # Unblock the waiting tasks


class TestRateLimiterRegistry:
    @pytest.mark.timeout(1)
    @pytest.mark.parametrize("target", [(), ("llm",), ("llm", "litellm", "openai")])
    async def test_throttle_return_empty_rate_limiter_if_no_rate_limiters_registered(
        self, target: tuple[str, ...]
    ):
        rate_limiter = throttle(*target)
        assert isinstance(rate_limiter, CompositeRateLimiter)
        assert len(rate_limiter.rate_limiters) == 0

        # ensure the rate limiter works as expected
        async def throttle_task(rate_limiter: RateLimiter) -> None:
            async with rate_limiter as details:
                assert details is NO_WAIT_TIME
                assert details.wait_time == 0
                await asyncio.sleep(1e-3)

        rate_limiter = throttle()
        async with asyncio.TaskGroup() as tg:
            for _ in range(10):
                tg.create_task(throttle_task(rate_limiter))

    @pytest.mark.parametrize(
        "target,expected_local",
        [
            (("llm",), True),
            (("llm", "openai"), True),
            (("embedding",), False),
        ],
    )
    async def test_throttle_global_rate_limiter_is_always_provided_first(
        self, target: tuple[str, ...], expected_local: bool
    ):
        with scoped_limiter(RateLimiter.rpm(100)) as global_policy:
            with scoped_limiter(RateLimiter.rpm(10), "llm") as local_policy:
                effective_limiter = throttle(*target)
                assert effective_limiter == (
                    CompositeRateLimiter(global_policy, local_policy)
                    if expected_local
                    else CompositeRateLimiter(global_policy)
                )

    @pytest.mark.parametrize(
        "input_target,expected_local",
        [
            (("llm",), False),
            (("llm", "openai"), False),
            (("llm", "openai", "gpt-5.2"), True),
            (("llm", "openai", "gpt-5.2", "2026-02-05"), True),
            (("llm", "openai", "gpt-5.2-codex"), False),
            (("embedding", "openai"), False),
            ((), False),
        ],
    )
    async def test_throttle_find_local_rate_limiter_when_global_rate_limiter_is_not_provided(
        self, input_target, expected_local
    ):
        with scoped_limiter(
            RateLimiter.rpm(10), "llm", "openai", "gpt-5.2"
        ) as local_policy:
            effective_limiter = throttle(*input_target)
            assert (
                effective_limiter == CompositeRateLimiter(local_policy)
                if expected_local
                else CompositeRateLimiter()
            ), effective_limiter.rate_limiters
