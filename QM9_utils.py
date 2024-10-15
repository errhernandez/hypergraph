
import re

import numpy as np

units = {}
units['A'] = 'GHz'
units['B'] = 'GHz'
units['C'] = 'GHz'
units['mu'] = 'D'
units['alpha'] = 'Bohr^3'
units['ehomo'] = 'Ha'
units['elumo'] = 'Ha'
units['egap'] = 'Ha'
units['R2'] = 'Bohr^2'
units['zpve'] = 'Ha'
units['U0'] = 'Ha'
units['U'] = 'Ha'
units['H'] = 'Ha'
units['G'] = 'Ha'
units['Cv'] = 'cal/mol/K'
units['freq'] = 'cm-1'

atom_ref = dict(
   H=-0.500273, C=-37.846772, N=-54.583861, O=-75.064579, F=-99.718730
)

def read_QM9_structure(
       file_name: str 
    ) -> tuple[int, int, list[str], float, dict[str, float]]:


    """

    This function opens a file of the GDB-9 database and processes it,
    returning the molecule structure in xyz format, a molecule identifier
    (tag), and a vector containing the entire list of molecular properties

    Args:

    :param str file_name: filename containing the molecular information

    :return: molecule_id (int): integer identifying the molecule number
        in the database n_atoms (int): number of atoms in the molecule
        species (list[str]): the species of each atom (len = n_atoms)
        coordinates (np.array(float)[n_atoms,3]): atomic positions
        properties (np.array(float)[:]): molecular properties, see
        database docummentation charge (np.array(float)[n_atoms]):
        Mulliken charges of atoms
    :rtype: tuple[int, int, list[str], float, float, float]

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

    # rather than returning a vector of properties, we will return 
    # a dictionary for ease of use

    prop_dict = {}
    prop_dict['A'] = molecular_data[0]
    prop_dict['B'] = molecular_data[1]
    prop_dict['C'] = molecular_data[2]
    prop_dict['mu'] = molecular_data[3]
    prop_dict['alpha'] = molecular_data[4]
    prop_dict['ehomo'] = molecular_data[5]
    prop_dict['elumo'] = molecular_data[6]
    prop_dict['egap'] = molecular_data[7]
    prop_dict['R2'] = molecular_data[8]
    prop_dict['zpve'] = molecular_data[9]
    prop_dict['U0'] = molecular_data[10]
    prop_dict['U'] = molecular_data[11]
    prop_dict['H'] = molecular_data[12]
    prop_dict['G'] = molecular_data[13]
    prop_dict['Cv'] = molecular_data[14]
    prop_dict['freq'] = frequencies

    return molecule_id, n_atoms, species, coordinates, prop_dict

def write_QM9_structure(
        file_name: str,
        molecule_id: int, 
        n_atoms: int,
        species: list[str],
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

       
