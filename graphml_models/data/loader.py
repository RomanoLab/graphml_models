import warnings

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset

graphs = [
    'ogbn-mag',
    'ogbl-biokg',
    ''
]

def load_graph(graph_name, format='pyg'):
    """
    Load a heterogeneous graph in a given format.

    Parameters
    ----------
    graph_name : str
        Name of a dataset as specified in OGB. If `None`, a warning will be raised and the
        available graph datasets will be listed.
    format : str, default='pyg'
        Output format of the graph. Options are ['pyg', 'dgl'].
    """
    if graph_name is None:
        warnings.warn("No graph name provided - listing available graphs instead.")
    else:
        # Make sure a valid graph is given
        if not graph_name in graphs:
            raise ValueError(f"Error - {graph_name} is not the name of an available graph.")
        
        task_abbrev = graph_name.split('-')[0][-1]
        if task_abbrev == 'n':
            dataset = PygNodePropPredDataset(name = graph_name)
        elif task_abbrev == 'l':
            dataset = PygLinkPropPredDataset(name = graph_name)
        elif task_abbrev == 'g':
            dataset = PygGraphPropPredDataset(name = graph_name)
        else:
            raise RuntimeError("Error - Couldn't parse graph type.")
        
        return dataset