import numpy as np
import pytest
from minisom_representation import SomRepresentation

def test_unfitted_access_raises():
    som_rep = SomRepresentation(d1=4, d2=4)
    with pytest.raises(AssertionError, match="not been fitted yet"):
        _ = som_rep.som
    with pytest.raises(AssertionError, match="not been fitted yet"):
        _ = som_rep.distance_map
    with pytest.raises(AssertionError, match="not been fitted yet"):
        _ = som_rep.activation_map
    with pytest.raises(AssertionError, match="not been fitted yet"):
        _ = som_rep.quantization_error
    with pytest.raises(AssertionError, match="not been fitted yet"):
        _ = som_rep.topographic_error
    with pytest.raises(AssertionError, match="not been fitted yet"):
        _ = som_rep.unique_b2mu_edges_counts_distances

def test_with_derived_params():
    np.random.seed(42)
    X = np.random.randn(30, 3)
    som_rep = SomRepresentation.with_derived_params(X, initial_sigma_factor=4.0, random_seed=42, verbose=True)
    
    # set params
    assert som_rep.d1 > 0
    assert som_rep.d2 > 0
    assert som_rep.sigma == float(max(som_rep.d1, som_rep.d2) / 4.0)
    assert som_rep.random_seed == 42
    assert som_rep.verbose == True

    # default params
    assert som_rep.topology == "rectangular"
    assert som_rep.learning_rate == .5
    assert som_rep.decay_function == "asymptotic_decay"
    assert som_rep.sigma_decay_function == "asymptotic_decay"
    assert som_rep.neighborhood_function == "gaussian"
    assert som_rep.activation_distance == "euclidean"
    assert som_rep.distance_map_scaling == "mean"

def test_fit_online_and_properties():
    np.random.seed(42)
    X = np.random.randn(30, 3)
    som_rep = SomRepresentation(d1=4, d2=4, random_seed=42)
    som_rep.fit_online(X, num_iteration=5)

    assert som_rep.som is not None
    assert som_rep.distance_map.shape == (4, 4)
    assert som_rep.activation_map.shape == (4, 4)
    assert som_rep.quantization_error >= 0.0
    assert 0.0 <= som_rep.topographic_error <= 1.0

    assert hasattr(som_rep, "fit_type_")
    assert hasattr(som_rep, "fitted_")
    assert hasattr(som_rep, "unique_b2mu_edges_")
    assert hasattr(som_rep, "unique_b2mu_counts_")
    assert hasattr(som_rep, "unique_b2mu_distances_")

    assert som_rep.fit_type_ == "online"
    assert som_rep.fitted_ is True
    assert som_rep.unique_b2mu_edges_ is not None
    assert som_rep.unique_b2mu_counts_ is not None
    assert som_rep.unique_b2mu_distances_ is not None

    edges, counts, distances = som_rep.unique_b2mu_edges_counts_distances
    assert len(edges) == len(counts) == len(distances)

def test_fit_offline_and_properties():
    np.random.seed(42)
    X = np.random.randn(30, 3)
    som_rep = SomRepresentation(d1=4, d2=4, random_seed=42)
    som_rep.fit_offline(X, num_iteration=5)

    assert som_rep.som is not None
    assert som_rep.distance_map.shape == (4, 4)
    assert som_rep.activation_map.shape == (4, 4)
    assert som_rep.quantization_error >= 0.0
    assert 0.0 <= som_rep.topographic_error <= 1.0

    assert hasattr(som_rep, "fit_type_")
    assert hasattr(som_rep, "fitted_")
    assert hasattr(som_rep, "unique_b2mu_edges_")
    assert hasattr(som_rep, "unique_b2mu_counts_")
    assert hasattr(som_rep, "unique_b2mu_distances_")

    assert som_rep.fit_type_ == "offline"
    assert som_rep.fitted_ is True
    assert som_rep.unique_b2mu_edges_ is not None
    assert som_rep.unique_b2mu_counts_ is not None
    assert som_rep.unique_b2mu_distances_ is not None

    edges, counts, distances = som_rep.unique_b2mu_edges_counts_distances
    assert len(edges) == len(counts) == len(distances)
