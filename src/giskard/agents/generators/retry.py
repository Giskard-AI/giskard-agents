import sys

from pydantic import BaseModel, Field

MAX_WAIT_SECONDS = sys.maxsize / 2


class RetryPolicy(BaseModel):
    """Configuration for retrying generator completions.

    The policy is applied by `WithRetryPolicy._complete` using Tenacity's
    exponential backoff. The retry condition itself is delegated to the
    generator's `_should_retry` implementation.

    Attributes
    ----------
    max_retries : int
        Maximum number of attempts, including the initial one.
    base_delay : float
        Base delay (in seconds) used as the exponential backoff multiplier.
    max_delay : float | None
        Maximum delay (in seconds) between retries. When `None`, a large
        sentinel value is used to avoid capping.
    """

    max_retries: int = Field(default=3)
    base_delay: float = Field(default=1.0)
    max_delay: float | None = Field(default=None)
