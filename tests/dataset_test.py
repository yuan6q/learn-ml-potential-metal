from src.data_processing.dataset import MDPotentialDataset

traj_files = [
    "data/raw/cu_nvt_300K.traj",
    "data/raw/cu_nvt_600K.traj",
    "data/raw/cu_nvt_900K.traj"
]

dataset = MDPotentialDataset(traj_files)
dataset.load()

print("Number of frames:", len(dataset))
print("Energy range:", dataset.get_energies().min(), dataset.get_energies().max())
print("Forces shape:", dataset.get_forces().shape)
