
import re
from typing import List, Tuple

import numpy as np

def read_QM9_structure(
       file_name: str 
    ) -> Tuple[int, int, List[str], float, float]:


    """

    This function opens a file of the GDB-9 database and processes it,
    returning the molecule structure in xyz format, a molecule identifier
    (tag), and a vector containing the entire list of molecular properties

    Args:

    :param str file_name: filename containing the molecular information

    :return: molecule_id (int): integer identifying the molecule number
        in the database n_atoms (int): number of atoms in the molecule
        species (List[str]): the species of each atom (len = n_atoms)
        coordinates (np.array(float)[n_atoms,3]): atomic positions
        properties (np.array(float)[:]): molecular properties, see
        database docummentation charge (np.array(float)[n_atoms]):
        Mulliken charges of atoms
    :rtype: Tuple[int, int, List[str], float, float, float]

    """

    with open(file_name, "r") as file_in:
        lines = file_in.readlines()

    n_atoms = int(lines[0])  # number of atoms is specified in 1st line

    words = lines[1].split()

    molecule_id = int(words[1])

    molecular_data = np.array(words[2:], dtype=float)

    species = []  # species label
    coordinates = np.zeros((n_atoms, 3), dtype=float)  # coordinates in Angstrom
    # charge = np.zeros((n_atoms), dtype=float)  # Mulliken charges (e)

    # below extract chemical labels, coordinates and charges

    m = 0

    for n in range(2, n_atoms + 2):

        line = re.sub(
            r"\*\^", "e", lines[n]
        )  # this prevents stupid exponential lines in the data base

        words = line.split()

        species.append(words[0])

        x = float(words[1])
        y = float(words[2])
        z = float(words[3])

        # c = float(words[4])

        coordinates[m, :] = x, y, z

        # charge[m] = c
    
        m += 1

    # finally obtain the vibrational frequencies, in cm^-1

    frequencies = np.array(lines[n_atoms + 2].split(), dtype=float)

    # we pack all the molecular data into a single array of properties

    properties = np.expand_dims(
       np.concatenate((molecular_data, frequencies)), axis=0
    )

    return molecule_id, n_atoms, species, coordinates, properties

def write_QM9_structure(
        file_name: str,
        molecule_id: int, 
        n_atoms: int,
        species: List[str],
        coordinates: float,
        properties: float 
    ) -> None:


    """

    This function does the opposite as read_QM9_structure, i.e. it writes
    an xyz file from the given data. It is only used for testing purposes, 
    such as checking forces, etc. Arguments are as above.

    """

    with open(file_name, "w") as file_out:

        txt = repr(n_atoms) + '\n'
        file_out.write(txt)

        n_degrees = 3 * n_atoms - 6
        _, n = properties.shape
        n -= n_degrees

        molecule_data = list(properties[0,:n])
        frequencies = list(properties[0,n:])

        txt = 'gdb ' + repr(molecule_id) + ' '

        for word in molecule_data:
               txt += repr(word) + ' '

        txt += '\n'

        file_out.write(txt)

        for n in range(n_atoms):

            txt = species[n] + ' ' + \
                    repr(coordinates[n,0]) + ' ' + \
                    repr(coordinates[n,1]) + ' ' + \
                    repr(coordinates[n,2]) + '\n'

            file_out.write(txt)

        txt = ''

        for word in frequencies:
            txt += repr(word) + '     '

        txt += '\n'

        file_out.write(txt)

       
