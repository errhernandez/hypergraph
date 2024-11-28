
from typing import Optional

from glob import glob
import pickle
import random
from torch.utils.data import Dataset

from hypergraph import HyperGraph

class HyperGraphDataSet(Dataset):

    """
    :Class:

    Data set class to load molecular hypergraph data
    """

    def __init__(
        self,
        database_dir: str = None,
        n_max_entries: int = None,
        seed: int = 42,
        file_extension: str = '.pkl',
        files: Optional[list[str]] = None
    ) -> None:

        """

        Args:

        :param str database_dir: the directory where the data files reside

        :param int n_max_entries: optionally used to limit the number of clusters
                  to consider; default is all

        :param int seed: initialises the random seed for choosing randomly
                  which data files to consider; the default ensures the
                  same sequence is used for the same number of files in
                  different runs

        :param str file_extension: the extension of files in the database; default = .xyz

        :files list[str] (optional): by default, with database_dir and file_extension this
                  class constructs a dataset with all the files (or n_max_entries if not None)
                  contained in database_dir. It might be desirable however to split the files
                  into two datasets (e.g. training and validation); in this case, the user
                  must provide a list of filenames (full relative path). 

        """

        self.database_dir = database_dir

        if files is None:

           filenames = database_dir + "/*"+file_extension

           files = glob(filenames)

        self.n_structures = len(files)

        """
        files contains a list of files, one for each item in
        the database if n_max_entries != None and is set to some integer
        value less than n_structures, then n_max_entries clusters are
        selected randomly for use.
        """

        if n_max_entries and n_max_entries < self.n_structures:

            self.n_structures = n_max_entries
            random.seed(seed)
            self.filenames = random.sample(files, n_max_entries)

        else:

            self.n_structures = len(files)
            self.filenames = files

    def __len__(self) -> int:
        """
        :return: the number of entries in the database
        :rtype: int 
        """

        return self.n_structures

    def __getitem__(self, idx: int) -> HyperGraph:

        """
        This function loads from file the corresponding data for entry
        idx in the database and returns the corresponding graph read
        from the file
  
        Args:

        :param int idx: the idx'th entry in the database
        :return: the idx'th graph in the database
        :rtype: HyperGraph

        """

        file_name = self.filenames[idx]

        with open(file_name, 'rb') as infile:
           hgraph = pickle.load(infile)

        return hgraph

    def get_filename(self, idx: int) -> str:

        """
        Returns the cluster data file name
        
        :param int idx: the idx'th entry in the database
        :return: the filename containing the structure data corresponding
           to the idx'th entry in the database
        :rtype: str

        """

        return self.filenames[idx]
