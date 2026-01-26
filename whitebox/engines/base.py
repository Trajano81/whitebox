"""Base class for plot engines."""

import abc


class PlotEngine(metaclass=abc.ABCMeta):
    """Abstract base class for plot engines."""

    @abc.abstractmethod
    def univariate_plot(self):
        """Generate a univariate plot."""
        pass

    @abc.abstractmethod
    def bivariate_plot(self):
        """Generate a bivariate plot."""
        pass
