"""
Definition of fcc Cu atomic system using a class-based builder.
"""

from ase.build import bulk


class CuFCCBuilder:
    """
    Builder class for fcc Cu supercell.
    """

    def __init__(
        self,
        lattice_constant: float = 3.615,
        supercell_size: tuple = (3, 3, 3)
    ):
        """
        Parameters
        ----------
        lattice_constant : float
            Lattice constant in angstrom.
        supercell_size : tuple of int
            Supercell repetition along (x, y, z).
        """
        self.lattice_constant = lattice_constant
        self.supercell_size = supercell_size

    def build(self):
        """
        Build and return an fcc Cu supercell.

        Returns
        -------
        ase.Atoms
            Periodic Cu supercell.
        """
        # Build fcc Cu unit cell
        cu_unit = bulk(
            name="Cu",
            crystalstructure="fcc",
            a=self.lattice_constant,
            cubic=True
        )

        # Build supercell
        cu_supercell = cu_unit.repeat(self.supercell_size)

        return cu_supercell
