import numpy as np
import matplotlib.pyplot as plt

from typing import Literal

from minisom_representation.som_representation import SomRepresentation
from minisom_representation.utils.attribute import extract_attributes

def plot_som_convergence_over_epochs(
        skeleton:SomRepresentation,
        X,
        fit_type:Literal["online", "offline"]="online",
        epoch_step=2,
        epoch_step_from=2,
        epoch_step_to=6,
        te_ceiling=0.1,
        figsize=(10, 3),
        verbose=True,
        show_fig=True
):

    epoch_axis = np.r_[1, np.arange(epoch_step_from, (epoch_step_to + 1), epoch_step)]
    qes, tes = [], []

    if verbose:
        print(f"Evaluating for epochs of {epoch_axis}...")

    params = extract_attributes(SomRepresentation, skeleton)

    for epoch in epoch_axis:
        if verbose:
            print(f"Training SOM for {epoch} epochs...")

        som_rep = SomRepresentation(**params)
        if fit_type == "online":
            som_rep.fit_online(X, num_iteration=epoch)
        else:
            som_rep.fit_offline(X, num_iteration=epoch)
        qes.append(som_rep.quantization_error)
        tes.append(som_rep.topographic_error)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    ax = ax1
    ax.scatter(epoch_axis, qes, color="cornflowerblue")
    ax.plot(epoch_axis, qes, color="cornflowerblue", label="QE")
    ax.set_xticks(epoch_axis)

    ax = ax2
    ax.scatter(epoch_axis, tes, color="coral")
    ax.plot(epoch_axis, tes, color="coral", label="TE")
    ax.hlines(te_ceiling, xmin=1, xmax=epoch_step_to, linestyle="dashed", color="grey", label="TE acceptance line")

    if show_fig:
        plt.legend()
        plt.show()

    return fig, qes, tes
