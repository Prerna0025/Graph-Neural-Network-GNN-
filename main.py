from omegaconf import OmegaConf
import networkx as nx
import numpy as np
from algorithms.WL import wl_1_algorithm
from data.Dataset_Loader import load_graph_from_dataset
from algorithms.Graph_plots import plot_graph
from encoders.wl_embedding import encode_labels,encode_onehot_labels,hex_to_normalized_values,train_graph_embeddings
from utils.logger import get_logger
logger = get_logger(__name__)

def main():
    # Load config from YAML file
    config = OmegaConf.load("configs/default.yaml")

    # Load graph and extract edge list
    G, edge_list = load_graph_from_dataset(config.dataset, config.graph_index)
    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Apply WL algorithm
    final_vector,all_iter_vector = wl_1_algorithm(G, hash_method=config.hash_method, max_itr=config.get("max_itr", 10))
    final_labels = nx.get_node_attributes(G, "label")

    # Decide which vector to use
    if config.get("all_iterations", False):
        vector_to_use = all_iter_vector
        logger.info("Using all iterations labels as feature vector")
    else:
        vector_to_use = final_vector
        logger.info("Using final labels only as feature vector")

    # Log and visualize feature vector
    for label, count in vector_to_use.items():
        logger.info(f"{label[:6]}...: {count}")
    vector = np.array(list(vector_to_use.values()))
    logger.info(f"Feature vector shape: {vector.shape}")
    plot_graph(
    G,
    labels=vector_to_use,
    title=f"{config.dataset} WL Graph",
    save_as="graph",
    method=config.mode  # This adds 'train', 'hex', etc. to the file name
)
    # Select encoding method
    if config.encoding_type == "onehot":
        logger.info("Using One-Hot Encoding")
        encode_onehot_labels(final_labels)
    elif config.encoding_type == "IntegerMapping":
        logger.info("Using Label Encoding")
        encode_labels(final_labels)
    elif config.encoding_type == "NormalizedValue":
        logger.info("Using Label Encoding")
        hex_to_normalized_values(final_labels)
    elif config.encoding_type == "DenseVector":
        logger.info("Using Dense Vector Training")
        train_graph_embeddings(
            final_labels=final_labels,
            edge_list=edge_list,
            embedding_dim=config.get("embedding_dim", 32),
            epochs=config.get("epochs", 50),
            save_path=f"outputs/wl_dense_embeddings_{config.hash_method}.csv"
        )


if __name__ == "__main__":
    main()
