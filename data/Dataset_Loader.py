import networkx as nx
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from utils.logger import get_logger
logger = get_logger(__name__)
'''
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
'''
def load_graph_from_dataset(name: str, index: int = 0):
    """
    Load a graph either from a synthetic example or from a TUDataset.

    Parameters:
    - name (str): Dataset name (e.g., 'PROTEINS', 'SYNTHETIC')
    - index (int): Index of the graph in the dataset to load

    Returns:
    - G (networkx.Graph): The loaded graph
    - edge_list (List[Tuple[int, int]]): List of edges
    """
    logger.info("Loading graph from dataset: %s (index: %d)", name, index)

    if name.upper() == "SYNTHETIC":
        G = nx.Graph()
        G.add_edges_from([
            (0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5),
            (3, 6), (6, 7), (4, 8), (5, 8), (7, 8)
        ])
        edge_list = list(G.edges())
        logger.info("Loaded synthetic graph with %d nodes and %d edges.", G.number_of_nodes(), G.number_of_edges())
        return G, edge_list

    try:
        dataset = TUDataset(root="data", name=name)
        pyg_graph = dataset[index]
        edge_index = pyg_graph.edge_index
        edge_list = edge_index.t().tolist()

        G = to_networkx(pyg_graph, to_undirected=True)

        if hasattr(pyg_graph, 'x') and pyg_graph.x is not None:
            logger.info("Node features shape: %s", pyg_graph.x.shape)
            logger.debug(torch.unique(pyg_graph.x, dim=0))
        else:
            logger.warning("Node features not present in the graph.")

        logger.info("Loaded %s graph with %d nodes and %d edges.", name, G.number_of_nodes(), len(edge_list))
        return G, edge_list

    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", name, str(e))
        raise
