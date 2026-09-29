<img width="30%" alt="Ophelia R&D logo" src="assets/ophelia_rnd_logo.png" />

---

[![PyPI version](https://img.shields.io/pypi/v/minisom-representation.svg)](https://pypi.org/project/minisom-representation/)

# MiniSom Representation

An advanced wrapper around [MiniSom](https://github.com/JustGlowing/minisom). 

It provides utilities to help with hyperparameter selection, convergence inspection, and BMU connection extraction for Self-Organizing Maps.

## Installation

```bash
pip install minisom-representation
```

## Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from minisom_representation import SomRepresentation

# 1. Prepare and scale data
X, _ = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# 2. Initialize with derived hyperparameters and fit
som_rep = SomRepresentation.with_derived_params(X_scaled, initial_sigma_factor=2.0, distance_map_scaling="mean", random_seed=42)
som_rep.fit_online(X_scaled, num_iteration=20, use_epochs=True, random_order=True)

# 3. Access MiniSom object, matrix properties, and error metrics
som = som_rep.som                              # Underlying MiniSom instance
distance_map = som_rep.distance_map            # Distance map (U-Matrix)
activation_map = som_rep.activation_map        # Activation response map
qe = som_rep.quantization_error                # Quantization Error (QE)
te = som_rep.topographic_error                 # Topographic Error (TE)
```

> [!TIP]
> Check out the [examples](examples/) folder for detailed usage.


