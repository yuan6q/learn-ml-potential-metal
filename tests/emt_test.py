from ase.calculators.emt import EMT
from systems.cu_fcc import CuFCCBuilder

# 构建 Cu 超胞
builder = CuFCCBuilder()
atoms = builder.build()

# 绑定 EMT 势
atoms.calc = EMT()

# 计算能量与力
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print("Total energy (eV):", energy)
print("Forces shape:", forces.shape)
print("Max force (eV/Å):", forces.max())
