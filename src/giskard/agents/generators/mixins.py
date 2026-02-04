from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import AsyncContextManager

import tenacity as t
from pydantic import BaseModel, Field, field_validator

from ..chat import Message
from ..rate_limiter import RateLimiter, get_rate_limiter
from .base import GenerationParams, Response
from .retry import MAX_WAIT_SECONDS, RetryPolicy


class WithRateLimiter(BaseModel):
    """Adds a rate limiter to the generator."""

    rate_limiter: RateLimiter | None = Field(default=None, validate_default=True)

    @field_validator("rate_limiter", mode="before")
    def _validate_rate_limiter(cls, v: RateLimiter | str | None) -> RateLimiter | None:
        # Supported singleton semantics are implemented at the container level:
        # - If a string is provided, it must already exist in the registry.
        # - If a dict is provided (e.g. from JSON deserialization), we reuse an
        #   already-registered instance when possible, otherwise we let Pydantic
        #   create a new RateLimiter from the dict.
        if v is None or isinstance(v, RateLimiter):
            return v

        if isinstance(v, str):
            return get_rate_limiter(v)

        if isinstance(v, dict):
            rate_limiter_id = v.get("rate_limiter_id")
            if rate_limiter_id:
                try:
                    return get_rate_limiter(rate_limiter_id)
                except ValueError:
                    return v  # let Pydantic create & register a new instance
            return v

        return v

    def _rate_limiter_context(self) -> AsyncContextManager:
        if self.rate_limiter is None:
            return nullcontext()

        return self.rate_limiter.throttle()


class WithRetryPolicy(BaseModel, ABC):
    """Adds retry behavior to generator completions.

    The policy is evaluated by `_complete` using Tenacity. To customize retry
    behavior, implement `_should_retry` in your generator and return `True`
    only for errors that are safe to retry (e.g., transient HTTP or rate-limit
    failures).

    Notes
    -----
    Custom generators should implement two methods:

    - `_complete_once` to perform a single request without retry logic.
    - `_should_retry` to decide whether a given exception should be retried.

    Avoid nested retries: callers should not wrap a generator that already
    includes `WithRetryPolicy` inside another retry mechanism, as this can
    multiply delays and obscure failure handling.
    """

    retry_policy: RetryPolicy | None = Field(default=RetryPolicy(max_retries=3))

    @abstractmethod
    def _should_retry(self, err: Exception) -> bool:
        """Return whether the exception should trigger a retry.

        Parameters
        ----------
        err : Exception
            The exception raised by `_complete_once`.

        Returns
        -------
        bool
            `True` if the request should be retried, otherwise `False`.
        """
        ...

    @abstractmethod
    async def _complete_once(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        """Complete a single request without retry logic.

        This method should be implemented by concrete generators to provide
        the actual completion logic. The retry policy will be applied by
        the _complete method.

        Parameters
        ----------
        messages : list[Message]
            List of messages to send to the model.
        params : GenerationParams | None
            Parameters for the generation.

        Returns
        -------
        Response
            The model's response.
        """
        ...

    def with_retries(
        self,
        max_retries: int,
        *,
        base_delay: float | None = None,
        max_delay: float | None = None,
    ) -> "WithRetryPolicy":
        current_policy = (
            self.retry_policy.model_dump(exclude_unset=True)
            if self.retry_policy is not None
            else {}
        )

        patch = {
            "max_retries": max_retries,
            "base_delay": base_delay,
            "max_delay": max_delay,
        }

        new_policy = current_policy | {k: v for k, v in patch.items() if v is not None}

        return self.model_copy(update={"retry_policy": RetryPolicy(**new_policy)})

    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        if self.retry_policy is None:
            return await self._complete_once(messages, params)

        retrier = t.AsyncRetrying(
            stop=t.stop_after_attempt(self.retry_policy.max_retries),
            wait=t.wait_exponential(
                multiplier=self.retry_policy.base_delay,
                max=self.retry_policy.max_delay or MAX_WAIT_SECONDS,
            ),
            retry=self._tenacity_retry_condition,
            before_sleep=self._tenacity_before_sleep,
            reraise=True,
        )

        return await retrier(self._complete_once, messages, params)

    def _tenacity_retry_condition(self, retry_state: t.RetryCallState) -> bool:
        return self._should_retry(retry_state.outcome.exception())

    def _tenacity_before_sleep(self, retry_state: t.RetryCallState) -> None:
        pass
