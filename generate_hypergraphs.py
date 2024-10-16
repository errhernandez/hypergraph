
from collections.abc import Generator
from pathlib import Path
import pickle

from atomic_structure_hypergraphs import AtomicStructureHyperGraphs
from hypergraph import HyperGraph

def generate_hypergraphs(graphs: AtomicStructureHyperGraphs,
        files: Generator[Path],
        target_dir: Path,
        output_file_ext: str = '.pkl') -> None:

    """
    This function uses an object of type AtomicStructureHyperGraphs to 
    create hyper-graphs (instances of HyperGraph) from molecular
    information contained in the given files (typically in xyz format) and 
    stores the resulting graphs as pickled files in the target directory
  
    Args:

    :param graphs: an instance of AtomicStructureHyperGraphs
                   used to transform molecular
                   structural information into a PyG Data object (a graph) 
    :param files: a Generator object yielding the the paths of files containing 
                   the molecular information for a given data-base.
    :param target_dir: the directory path where to place the pickled graphs

    """

    for file in files:

        words = file.parts

        input_file = words[-1]

        words = input_file.split('.')
 
        graph_file = str(target_dir) + '/' + words[0] + output_file_ext

        file_name = str(file)

        structure = graphs.structure2graph(file_name)

        with open( graph_file, 'wb' ) as outfile:
           pickle.dump(structure, outfile)
