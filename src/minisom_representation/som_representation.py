import numpy as np

from minisom import MiniSom
from minisom_representation.som_hyperparameter import calc_som_hyparams
from typing import Literal

class SomRepresentation():

    def __init__(self, d1, d2, sigma=1.0, topology="rectangular", learning_rate=0.5, num_iteration=20,
                    decay_function="asymptotic_decay", sigma_decay_function="asymptotic_decay",
                    neighborhood_function='gaussian', activation_distance='euclidean',
                    distance_map_scaling:Literal["mean", "sum"]="mean",
                    random_seed=None, verbose=False):
        self.d1 = d1
        self.d2 = d2
        self.sigma = sigma
        self.topology = topology
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.decay_function = decay_function
        self.sigma_decay_function = sigma_decay_function
        self.neighborhood_function = neighborhood_function
        self.activation_distance = activation_distance
        self.distance_map_scaling = distance_map_scaling
        self.random_seed = random_seed
        self.verbose = verbose

    @classmethod
    def with_derived_params(cls, X, **kwargs):
        """Derives grid shape (d1, d2) and sigma from X, allowing explicit keyword overrides."""
        derived_params = calc_som_hyparams(X, verbose=kwargs.get("verbose", False))
        return cls(**{**derived_params, **kwargs})

    @property
    def som(self):
        self.__require_fitted()
        return self.som_

    @property
    def distance_map(self):
        self.__require_fitted()
        return self.distance_map_

    @property
    def activation_map(self):
        self.__require_fitted()
        return self.activation_map_.astype(int)

    @property
    def unique_b2mu_edges_counts_distances(self):
        """Returns a tuple of unique edges (a1 b1 a2 b2), counts, and distances"""
        self.__require_fitted()
        return self.unique_b2mu_edges_, self.unique_b2mu_counts_, self.unique_b2mu_distances_

    @property
    def quantization_error(self):
        self.__require_fitted()
        return self.QE_

    @property
    def topographic_error(self):
        self.__require_fitted()
        return self.TE_

    def fit(self, X):

        som_hyperparams = {
            "input_len": X.shape[1],
            "x": self.d1,
            "y": self.d2,
            "sigma": self.sigma,
            "topology": self.topology,
            "learning_rate": self.learning_rate,
            "decay_function": self.decay_function,
            "sigma_decay_function": self.sigma_decay_function,
            "neighborhood_function": self.neighborhood_function,
            "activation_distance": self.activation_distance,
            "random_seed": self.random_seed
        }

        som_train_hyperparams = {
            "num_iteration": self.num_iteration,
            "verbose": self.verbose
        }

        som = MiniSom(**som_hyperparams)
        som.random_weights_init(X)
        som.train_batch_offline(X, **som_train_hyperparams)

        self.node_weights_ = som.get_weights().copy()
        self.component_size_ = self.node_weights_.shape[2]
        self.distance_map_ = som.distance_map(scaling=self.distance_map_scaling).copy()
        self.activation_map_ = som.activation_response(X).copy()
        self.lattice_shape_ = self.distance_map_.shape
        self.rows_, self.cols_ = self.distance_map_.shape

        b2mu_flat_inds_ = np.argsort(som._distance_from_weights(X), axis=1)[:, :2]
        b2mu_x_inds, b2mu_y_inds = np.unravel_index(b2mu_flat_inds_, self.lattice_shape_)
        b2mu_flat_inds_distance_ = np.linalg.norm(
            np.hstack([np.diff(b2mu_x_inds), np.diff(b2mu_y_inds)]),
            axis=1
        )

        bmus_1 = np.column_stack([b2mu_x_inds[:, 0], b2mu_y_inds[:, 0]])
        bmus_2 = np.column_stack([b2mu_x_inds[:, 1], b2mu_y_inds[:, 1]])
        b2mu_edges = np.sort(np.stack([bmus_1, bmus_2], axis=1), axis=1).reshape(-1, 4)
        unique_b2mu_edges, unique_b2mu_flat_inds, unique_b2mu_counts = np.unique(
            b2mu_edges, axis=0, return_index=True, return_counts=True
        )
        unique_b2mu_distances = b2mu_flat_inds_distance_[unique_b2mu_flat_inds]

        self.unique_b2mu_edges_ = unique_b2mu_edges
        self.unique_b2mu_counts_ = unique_b2mu_counts
        self.unique_b2mu_distances_ = unique_b2mu_distances

        self.QE_ = som.quantization_error(X)
        self.TE_ = som.topographic_error(X)

        self.som_ = som
        self.fitted_ = True

        if self.verbose:
            print("\n", "An SOM representation has been fitted as follows:")
            print("-------------------------------------------------------", "\n")
            print("Hyperparameters of SOM:", "\n")
            print({
                **som_hyperparams,
                **som_train_hyperparams
            }, "\n")

            print("Quality of SOM:", "\n")
            print(f"Quantization Error (QE):\t{self.QE_}")
            print(f"Topographic Error (TE): \t{self.TE_}")

        return self

    def __require_fitted(self):
        assert hasattr(self, "fitted_") and self.fitted_, "This SOM representation has not been fitted yet."
