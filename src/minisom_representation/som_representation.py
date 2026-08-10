from minisom import MiniSom
from minisom_representation.som_hyperparameter import calc_som_hyparams

class SomRepresentation():

    def __init__(self, d1, d2, sigma=1.0, topology="rectangular", learning_rate=0.5, num_iteration=20,
                    decay_function="asymptotic_decay", sigma_decay_function="asymptotic_decay",
                    neighborhood_function='gaussian', activation_distance='euclidean',
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
        self.random_seed = random_seed
        self.verbose = verbose

    @classmethod
    def with_derived_params(cls, X, random_seed=None, verbose=False, **kwargs):
        """Derives d1, d2, sigma from X, forwarding any extra overrides to __init__."""
        derived_params = calc_som_hyparams(X, verbose=verbose)
        return cls(random_seed=random_seed, verbose=verbose, **{**derived_params, **kwargs})

    @property
    def som(self):
        return self.som_

    @property
    def quantization_error(self):
        return self.QE_

    @property
    def topographic_error(self):
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

        QE = som.quantization_error(X)
        TE = som.topographic_error(X)

        self.som_ = som
        self.QE_ = QE
        self.TE_ = TE

        if self.verbose:
            print("\n", "An SOM representation has been fitted as follows:")
            print("-------------------------------------------------------", "\n")
            print("Hyperparameters of SOM:", "\n")
            print({
                **som_hyperparams,
                **som_train_hyperparams
            }, "\n")

            print("Quality of SOM:", "\n")
            print(f"Quantization Error (QE):\t{QE}")
            print(f"Topographic Error (TE): \t{TE}")

        return self
