from importlib.metadata import PackageNotFoundError, metadata

from minisom_representation.som_representation import SomRepresentation
from minisom_representation.som_convergence import plot_som_convergence_over_epochs
from minisom_representation.som_hyperparameter import *
from minisom_representation.som_hyperparameter import __all__ as som_hyperparameter_all

__version__ = "0.0.1"

try:
    _meta = metadata("minisom-representation")
    __description__ = _meta["Summary"]
except PackageNotFoundError:
    __description__ = ""

def describe():
    description = (
        "MiniSom Representation (minisom-representation)\n"
        "Description: {}\n"
        "Version: {}\n"
    ).format(__description__, __version__)

    print(description)

__all__ = ["__version__", "SomRepresentation", "plot_som_convergence_over_epochs"] + som_hyperparameter_all
