
from collections.abc import Iterator

from hypergraph import HyperGraph
from hypergraph_batch import hypergraph_batch
from hypergraph_dataset import HyperGraphDataSet

class HyperGraphDataLoader(Iterator):

    """
    This class is written in emulation of the pytorch dataloader class. It 
    creates an Iterator over batches of hypergraphs. Each batch is itself
    a hypergraph constructed by merging together batch_size hypergraphs, 
    while tracking the original hypergraph from which each node and hyperedge
    originated from (through batch_node_index and batch_hedge_index).

    """

    def __init__(self,
                 dataset: HyperGraphDataSet,
                 batch_size: int = 1,
                 return_last: bool = True
        ) -> None:
        """
        Create an instance of a hypergraph data-loader.

        Args:

          dataset (HyperGraphDataSet): the path to the dataset; this will be a 
               directory containing *.pkl files, each one of them containing a
               pickled hypergraph.

          batch_size (int): the number of hypergraphs in the batch; as explained above, 
               outwardly each batch will look like a single hypergraph, but it will
               in fact contain batch_size hypergraphs all packed together.

          return_last (bool=True): if the batch_size not an exact divisor of
               the length of the dataset, the last batch will have fewer hypergraphs
               than all the previous ones. In that case, the user can choose to 
               ignore the last (smaller) batch. By default the last smaller batch 
               will be returned.

        """

        self.dataset = dataset
        self.length = dataset.len()
        self.batch_size = batch_size
        self.return_last = return_last

        self.n_batches = int(self.length / self.batch_size)

        self.last_batch = False
        if self.length % self.batch_size > 0:
           self.last_batch = True
           self.n_last_batch = self.length % self.batch_size

        self.item_index = 0
        self.batch_index = 0

    def __next__(self) -> HyperGraph:

        if self.item_index == self.length:
         
           raise StopIteration

        hgraph_list = []

        n_item = 0

        while n_item < self.batch_size and self.item_index < self.length:

           hgraph_list.append(self.dataset.get(self.item_index))

           n_item += 1
           self.item_index += 1

        batch = hypergraph_batch(hgraph_list)

        if n_item == self.batch_size:

           return batch

        elif self.item_index == self.length:

           if self.return_last:

              return batch

