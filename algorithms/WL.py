import networkx as nx
from collections import defaultdict
import hashlib
from typing import Dict, List, Tuple
from utils.logger import get_logger
logger = get_logger(__name__)

#logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def hash_label(current_label: str, neighbor_labels: List[str], method: str = "md5") -> str:
    """
    Generate a hashed label by combining a node's label with its neighbors' labels.

    Args:
        current_label (str): Current label of the node.
        neighbor_labels (List[str]): Labels of the neighboring nodes.
        method (str): Hash method to use (default: 'md5').

    Returns:
        str: New hashed label.
    """
    combined = f"{current_label}-{'-'.join(sorted(neighbor_labels))}"
    hash_func = getattr(hashlib, method)
    return hash_func(combined.encode()).hexdigest()

def wl_1_algorithm(G: nx.Graph, hash_method: str = "md5", max_itr: int = 10) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Perform the Weisfeiler-Lehman (1-WL) label refinement algorithm on a graph.

    Args:
        G (nx.Graph): Input graph.
        hash_method (str): Hash function to use for label compression.
        max_itr (int): Maximum number of WL iterations.

    Returns:
        Tuple containing:
            - final_histogram (Dict[str, int]): Histogram of labels after final iteration.
            - all_iteration_histogram (Dict[str, int]): Histogram across all iterations.
    """
    logger.info(f"Starting 1-WL algorithm with max_itr={max_itr}, hash_method={hash_method}")

    if not nx.get_node_attributes(G, "label"):
        nx.set_node_attributes(G, "0", "label")
        logger.info("Initialized all node labels to '0'")

    all_iteration_histogram = defaultdict(int)
    final_histogram = defaultdict(int)

    for iteration in range(max_itr):
        logger.info(f"Iteration {iteration + 1}")
        current_labels = nx.get_node_attributes(G, "label")
        new_labels = {}

        for node in G.nodes():
            neighbor_labels = [current_labels[neighbor] for neighbor in G.neighbors(node)]
            new_label = hash_label(current_labels[node], neighbor_labels, method=hash_method)
            new_labels[node] = new_label

        if new_labels == current_labels:
            logger.info("Labels have converged. Stopping early.")
            break

        for label in new_labels.values():
            all_iteration_histogram[label] += 1

        nx.set_node_attributes(G, new_labels, "label")

    final_labels = nx.get_node_attributes(G, "label")
    for label in final_labels.values():
        final_histogram[label] += 1

    logger.info(f"Completed 1-WL. Unique final labels: {len(final_histogram)}")
    return dict(final_histogram), dict(all_iteration_histogram)
