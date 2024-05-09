import abc


class Plot_interface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def univariate_plot(self):
        pass

    @abc.abstractmethod
    def bivariate_plot(self):
        pass