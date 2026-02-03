import sys

from pydantic import BaseModel, Field

MAX_WAIT_SECONDS = sys.maxsize / 2


class RetryPolicy(BaseModel):
    """Adds a retry policy to the generator."""

    max_retries: int = Field(default=3)
    base_delay: float = Field(default=1.0)
    max_delay: float | None = Field(default=None)
