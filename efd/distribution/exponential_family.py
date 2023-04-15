from abc import ABC, abstractmethod
from math import exp
from typing import Generic, TypeVar

from efd.linear_algebra import ColumnVector, as_scalar


class Parameter(ABC):
    """Parameter for a distribution."""


Theta = TypeVar("Theta", bound=Parameter)


class ExponentialFamily(ABC, Generic[Theta]):
    """Exponential family of distribution."""

    @property
    @abstractmethod
    def parameter(self) -> Theta:
        """Parameter, not necessarily the natural parameter."""

    @property
    @abstractmethod
    def natural_parameter(self) -> ColumnVector:
        """Obtain the natural parameter: `eta`."""

    def density(self, x: ColumnVector) -> float:
        h = self.base_measure(x)
        eta = self.natural_parameter
        t = self.sufficient_statistics(x)
        a = self.log_partition(self.parameter)
        return h * exp(-a + as_scalar(eta.transpose() * t))

    @staticmethod
    @abstractmethod
    def sufficient_statistics(x: ColumnVector) -> ColumnVector:
        """Calculate the sufficient statistic: `T(x)`."""

    @staticmethod
    @abstractmethod
    def base_measure(x: ColumnVector) -> float:
        """Calculate the base measure: `h(x)`."""

    @staticmethod
    @abstractmethod
    def log_partition(theta: Theta) -> float:
        """Calculate the log-partition: `a(eta)`."""
