from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Molecule:
    atoms: List[str]
    x: np.array
    comment: str = ''

    def __post_init__(self):
        assert (
            len(self.atoms) == self.x.shape[0]
        ), "Incorrect number of atoms/coordinate"
        assert self.x.shape[1] == 3, "I want triplets!"
        assert all(
            isinstance(atom, str) for atom in self.atoms
        ), "I want a list of strings!"

    @classmethod
    def from_xyz(cls, fn):
        """Read xyz file into components"""
        with open(fn, "r") as f:
            line = next(f)
            natom = int(line)
            comment = next(f).rstrip("\n")
            atom_names = []
            geom = np.zeros((natom, 3), float)
            for i in range(natom):
                line = next(f).split()
                atom_names.append(str(line[0]))
                geom[i] = list(map(float, line[1:4]))
            # print(f"{natom=}\n{geom=}\n{atom_names=}\n{comment=}")
        return cls(comment=comment, atoms=atom_names, x=geom)

    @property
    def natoms(self):
        '''
        Molecule property corresponding to number of atoms (integer)
        '''
        return len(self.atoms)

    @property
    def nelectrons(self):
        '''
        Molecule property corresponding to number of electrons (integer)
        '''
        nelec = 0
        known_elements = {
                'H': 1,
                'C': 6,
                'N': 7,
                'O': 8,
                'F': 9,
                'P': 15,
                'S': 16,
                'Cl': 17,
                'Br': 35,
                'I': 53
            }
        for elem in self.atoms:
            assert (
                elem in known_elements
            ), f"There is an element I don't know {elem}, add it to the code"
            nelec += known_elements[elem]
        return nelec


if __name__ == "__main__":
    # m = Molecule(atoms=["C", "H"], x=np.arange(6).reshape(2, 3))
    m = Molecule.from_xyz('geom.xyz')
    nelec = m.nelectrons
