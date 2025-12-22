from data_processing.dataset import MDPotentialDataset
from data_processing.visualization import (
    plot_energy_histogram,
    plot_force_magnitude,
    plot_energy_vs_temperature
)
import numpy as np

traj_files = {
    300: "data/raw/cu_nvt_300K.traj",
    600: "data/raw/cu_nvt_600K.traj",
    900: "data/raw/cu_nvt_900K.traj"
}

energies_by_temp = {}
energy_stats = {}
all_forces = []

for temp, path in traj_files.items():
    dataset = MDPotentialDataset([path])
    dataset.load()

    energies = dataset.get_energies()
    energies_by_temp[temp] = energies

    energy_stats[temp] = {
        "mean": energies.mean(),
        "std": energies.std()
    }

    all_forces.append(dataset.get_forces())

all_forces = np.concatenate(all_forces, axis=0)

plot_energy_histogram(energies_by_temp)
plot_force_magnitude(all_forces)
plot_energy_vs_temperature(energy_stats)
