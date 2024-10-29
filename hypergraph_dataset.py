
from glob import glob
import pickle
import random

from hypergraph import HyperGraph

class HyperGraphDataSet:

    """
    :Class:

    Data set class to load molecular hypergraph data
    """

    def __init__(
        self,
        database_dir: str,
        nMaxEntries: int = None,
        seed: int = 42,
        transform: object = None,
        pre_transform: object = None,
        pre_filter: object = None,
        file_extension: str = '.pkl'
    ) -> None:

        """

        Args:

        :param str database_dir: the directory where the data files reside

        :param int nMaxEntries: optionally used to limit the number of clusters
                  to consider; default is all

        :param int seed: initialises the random seed for choosing randomly
                  which data files to consider; the default ensures the
                  same sequence is used for the same number of files in
                  different runs

        :param str file_extension: the extension of files in the database; default = .xyz

        """

        self.database_dir = database_dir

        filenames = database_dir + "/*"+file_extension

        files = glob(filenames)

        self.n_structures = len(files)

        """
        filenames contains a list of files, one for each item in
        the database if nMaxEntries != None and is set to some integer
        value less than n_structures, then nMaxEntries clusters are
        selected randomly for use.
        """

        if nMaxEntries and nMaxEntries < self.n_structures:

            self.n_structures = nMaxEntries
            random.seed(seed)
            self.filenames = random.sample(files, nMaxEntries)

        else:

            self.n_structures = len(files)
            self.filenames = files

    def len(self) -> int:
        """
        :return: the number of entries in the database
        :rtype: int 
        """

        return self.n_structures

    def get(self, idx: int) -> HyperGraph:

        """
        This function loads from file the corresponding data for entry
        idx in the database and returns the corresponding graph read
        from the file
  
        Args:

        :param int idx: the idx'th entry in the database
        :return: the idx'th graph in the database
        :rtype: HyperGraph

        """

        # if torch.is_tensor(idx):
        #   idx = idx.tolist()

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
