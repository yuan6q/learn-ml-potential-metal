"""
Minimal NVT MD simulation for fcc Cu using EMT potential.
"""

import os

from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase import units
from ase.io import Trajectory

from learnmlp.systems.cu_fcc import CuFCCBuilder

def run_nvt_md(
    temperature: float = 300.0,
    timestep_fs: float = 1.0,
    friction: float = 0.01,
    n_steps: int = 1000,
    output_dir: str = "data/raw"
):
    """
    Run a minimal NVT MD simulation.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.
    timestep_fs : float
        MD timestep in femtoseconds.
    friction : float
        Langevin friction coefficient (1/fs).
    n_steps : int
        Number of MD steps.
    output_dir : str
        Directory to save trajectory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build atomic system
    builder = CuFCCBuilder()
    atoms = builder.build()

    # Assign EMT calculator
    atoms.calc = EMT()

    # Set up MD
    dyn = Langevin(
        atoms,
        timestep_fs * units.fs,
        temperature_K=temperature,
        friction=friction
    )

    # Trajectory output
    traj_file = os.path.join(output_dir, f"cu_nvt_{int(temperature)}K.traj")
    traj = Trajectory(traj_file, "w", atoms)
    dyn.attach(traj.write, interval=1)

    # Run MD
    dyn.run(n_steps)

    print(f"MD finished. Trajectory saved to {traj_file}")


if __name__ == "__main__":
    temperatures = [300, 600, 900]

    for T in temperatures:
        print(f"Running MD at {T} K")
        run_nvt_md(
            temperature=T,
            timestep_fs=1.0,
            friction=0.01,
            n_steps=2000,   # 稍微加长，增强统计
            output_dir="data/raw"
        )
