"""
Dataset construction from MD trajectories.
"""

import os
import numpy as np
from ase.io import Trajectory


class MDPotentialDataset:
    """
    Dataset for machine learning interatomic potentials.
    """

    def __init__(self, traj_files):
        """
        Parameters
        ----------
        traj_files : list of str
            Paths to ASE trajectory files.
        """
        self.traj_files = traj_files
        self.data = []

    def load(self):
        """
        Load data from trajectory files.
        """
        for traj_file in self.traj_files:
            traj = Trajectory(traj_file)
            for atoms in traj:
                sample = {
                    "positions": atoms.get_positions(),
                    "atomic_numbers": atoms.get_atomic_numbers(),
                    "energy": atoms.get_potential_energy(),
                    "forces": atoms.get_forces(),
                    "cell": atoms.get_cell().array
                }
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def get_energies(self):
        return np.array([d["energy"] for d in self.data])

    def get_forces(self):
        return np.concatenate([d["forces"] for d in self.data], axis=0)


import torch
from torch.utils.data import Dataset
from ase.io import read


class TorchMDDataset(Dataset):
    """
    Minimal dataset wrapper for TorchMD-Net.
    """

    def __init__(self, traj_files):
        self.frames = []
        for traj in traj_files:
            self.frames.extend(read(traj, ":"))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        atoms = self.frames[idx]

        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
        forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)

        return {
            "pos": pos,
            "z": z,
            "y": energy,
            "dy": forces
        }

