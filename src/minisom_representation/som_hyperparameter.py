import numpy as np

__all__ = ["calc_recommended_total_node_count", "calc_recommended_lattice_side_ratio", "calc_recommended_lattice_sides", "calc_initial_sigma", "calc_som_hyparams"]

def calc_recommended_total_node_count(X):
    N = len(X)
    M = int(np.ceil(5 * np.sqrt(N)))
    return M

def calc_recommended_lattice_side_ratio(X):
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    lambda2, lambda1 = eigvals[-2:]
    eps = 1e-8
    lambda1 = max(float(lambda1), eps)
    lambda2 = max(float(lambda2), eps)
    ratio = np.sqrt(lambda1 / lambda2)
    return ratio

def calc_recommended_lattice_sides(X):
    M = calc_recommended_total_node_count(X)
    ratio = calc_recommended_lattice_side_ratio(X)
    height = int(np.ceil(np.sqrt(M / ratio)))
    width = int(np.ceil(ratio * height))
    return height, width

def calc_initial_sigma(d1, d2, factor=2.0):
    L = max(d1, d2)
    if L <= 1: return 1.0
    return np.round(L / factor, 2)

def calc_som_hyparams(X, initial_sigma_factor=2.0, verbose=False):
    total_node_count = calc_recommended_total_node_count(X)
    height, width = calc_recommended_lattice_sides(X)
    initial_sigma = calc_initial_sigma(height, width, initial_sigma_factor)

    hyperparams = {
        "d1": height,
        "d2": width,
        "sigma": initial_sigma,
    }

    if verbose:
        print("\n--------------")
        print(f"Based on {len(X)} instances, the recommended SOM hyperparameters are the following:")
        print("Total node count (M):\t", total_node_count)
        print("Recommended sides (d1 x d2):\t", f"{height} x {width}")
        print("Initial neighborhood radius (sigma):\t", initial_sigma)
        print("--------------\n")

    return hyperparams
