import torch
from torch.utils.data import DataLoader
from learnmlp.data_processing.dataset import TorchMDDataset
from learnmlp.model_training.minimal_schnet import SimpleSchNet


def train():
    trajs = [
        "data/raw/cu_nvt_300K.traj",
        "data/raw/cu_nvt_600K.traj",
        "data/raw/cu_nvt_900K.traj"
    ]

    dataset = TorchMDDataset(trajs)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = SimpleSchNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        loss_sum = 0.0
        for batch in loader:
            optimizer.zero_grad()

            energy, forces = model(
                batch["z"][0],
                batch["pos"][0]
            )

            loss_e = (energy - batch["y"][0]) ** 2
            loss_f = torch.mean((forces - batch["dy"][0]) ** 2)

            loss = loss_e + loss_f
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        print(f"Epoch {epoch:03d} | Loss {loss_sum:.6f}")


if __name__ == "__main__":
    train()
