from systems.cu_fcc import CuFCCBuilder

builder = CuFCCBuilder(
    lattice_constant=3.615,
    supercell_size=(3, 3, 3)
)

atoms = builder.build()

print("Number of atoms:", len(atoms))
print("Cell:\n", atoms.cell)
