from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Distribution(ABC, Generic[T]):
    @abstractmethod
    def sample(self) -> T:
        """Draw a random sample."""
