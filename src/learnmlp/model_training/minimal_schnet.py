import torch
import torch.nn as nn


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=50, cutoff=5.0):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(0, cutoff, n_rbf))
        self.gamma = nn.Parameter(torch.tensor(10.0))

    def forward(self, distances):
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff ** 2)


class SimpleSchNet(nn.Module):
    def __init__(self, n_atom_types=100, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, hidden_dim)
        self.rbf = GaussianRBF()
        self.filter_net = nn.Sequential(
            nn.Linear(50, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, pos):
        pos.requires_grad_(True)

        h = self.embedding(z)
        rij = pos.unsqueeze(1) - pos.unsqueeze(0)
        distances = torch.norm(rij + 1e-9, dim=-1)

        rbf = self.rbf(distances)
        filters = self.filter_net(rbf)

        h = h + torch.sum(filters * h.unsqueeze(1), dim=1)
        energy = self.dense(h).sum()

        forces = -torch.autograd.grad(
            energy, pos, create_graph=True
        )[0]

        return energy, forces
