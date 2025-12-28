import torch
from torch.utils.data import DataLoader
from torchmdnet.models.model import TorchMD_Net

from learnmlp.data_processing.torchmd_dataset import TorchMDDataset


def train():
    trajs = [
        "data/raw/cu_nvt_300K.traj",
        "data/raw/cu_nvt_600K.traj",
        "data/raw/cu_nvt_900K.traj"
    ]

    dataset = TorchMDDataset(trajs)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TorchMD_Net(
        representation="schnet",
        rbf_type="gauss",
        num_filters=64,
        num_interactions=3,
        cutoff=5.0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()

            energy, forces = model(
                z=batch["z"],
                pos=batch["pos"]
            )

            loss_e = torch.mean((energy.squeeze() - batch["y"]) ** 2)
            loss_f = torch.mean((forces - batch["dy"]) ** 2)

            loss = loss_e + loss_f
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Loss: {total_loss:.6f}")


if __name__ == "__main__":
    train()
