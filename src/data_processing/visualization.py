"""
Visualization utilities for ML potential datasets.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_energy_histogram(energies_by_temp, bins=50):
    """
    Plot energy distribution for different temperatures.
    """
    plt.figure()
    for temp, energies in energies_by_temp.items():
        plt.hist(
            energies,
            bins=bins,
            density=True,
            alpha=0.5,
            label=f"{temp} K"
        )
    plt.xlabel("Total Energy (eV)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_force_magnitude(forces, bins=100):
    """
    Plot force magnitude distribution.
    """
    magnitudes = np.linalg.norm(forces, axis=1)

    plt.figure()
    plt.hist(
        magnitudes,
        bins=bins,
        density=True
    )
    plt.xlabel("Force Magnitude (eV/Å)")
    plt.ylabel("Probability Density")
    plt.tight_layout()
    plt.show()


def plot_energy_vs_temperature(stats):
    """
    Plot mean energy with error bars.
    """
    temps = sorted(stats.keys())
    means = [stats[t]["mean"] for t in temps]
    stds = [stats[t]["std"] for t in temps]

    plt.figure()
    plt.errorbar(
        temps,
        means,
        yerr=stds,
        fmt="o-"
    )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Total Energy (eV)")
    plt.tight_layout()
    plt.show()
